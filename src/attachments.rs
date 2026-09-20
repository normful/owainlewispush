//! Provider-neutral inbound attachment files (PLAN §5).
//!
//! Images keep their own module (`crate::image`) because they are validated as
//! decodable JPEG/PNG/WebP content. Everything else the agent should see — a
//! PDF, a CSV, a tarball — is an opaque file: push never inspects the bytes, it
//! only writes them to a temporary file and hands the path to the backend.

use std::fs::{DirBuilder, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use uuid::Uuid;

pub const MAX_FILE_COUNT: usize = 4;
pub const MAX_FILE_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub struct DownloadedFile {
    pub bytes: Vec<u8>,
    pub filename: String,
}

/// Temporary files for one inbound message, removed when dropped.
pub struct PreparedFiles {
    directory: PathBuf,
    paths: Vec<PathBuf>,
}

impl PreparedFiles {
    pub fn create(cache_dir: &Path, files: Vec<DownloadedFile>) -> Result<Self> {
        if files.is_empty() {
            return Ok(Self {
                directory: PathBuf::new(),
                paths: Vec::new(),
            });
        }
        if files.len() > MAX_FILE_COUNT {
            bail!("message has more than {MAX_FILE_COUNT} files");
        }
        let total = files
            .iter()
            .try_fold(0usize, |total, file| total.checked_add(file.bytes.len()))
            .context("attachment size overflow")?;
        if total > MAX_FILE_BYTES {
            bail!("attachments exceed the 8 MiB limit");
        }

        std::fs::create_dir_all(cache_dir).with_context(|| {
            format!("create attachment cache directory {}", cache_dir.display())
        })?;
        crate::util::restrict_permissions(cache_dir, true).with_context(|| {
            format!(
                "restrict attachment cache directory {}",
                cache_dir.display()
            )
        })?;
        let directory = cache_dir.join(format!("files-{}", Uuid::new_v4()));
        DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .with_context(|| format!("create temporary file directory {}", directory.display()))?;

        let mut prepared = Self {
            directory,
            paths: Vec::with_capacity(files.len()),
        };
        for (index, file) in files.into_iter().enumerate() {
            // The display name is attacker-controlled, so it never becomes a
            // path component: only the basename survives, and an index prefix
            // keeps two same-named files distinct.
            let basename = safe_basename(&file.filename).unwrap_or("attachment");
            let path = prepared.directory.join(format!("{}-{basename}", index + 1));
            let mut handle = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
                .with_context(|| format!("create temporary file {}", path.display()))?;
            prepared.paths.push(path.clone());
            handle
                .write_all(&file.bytes)
                .with_context(|| format!("write temporary file {}", path.display()))?;
            handle
                .sync_all()
                .with_context(|| format!("sync temporary file {}", path.display()))?;
        }
        Ok(prepared)
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
}

impl Drop for PreparedFiles {
    fn drop(&mut self) {
        for path in &self.paths {
            let _ = std::fs::remove_file(path);
        }
        if !self.directory.as_os_str().is_empty() {
            let _ = std::fs::remove_dir(&self.directory);
        }
    }
}

/// Reduces a provider-supplied filename to a single safe path component.
fn safe_basename(filename: &str) -> Option<&str> {
    let basename = Path::new(filename.trim()).file_name()?.to_str()?;
    if basename.is_empty() || basename.starts_with('.') {
        return None;
    }
    Some(basename)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::temp_dir;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn writes_files_under_sanitized_names_and_removes_them() {
        let cache = temp_dir("prepared-files");
        let prepared = PreparedFiles::create(
            &cache,
            vec![
                DownloadedFile {
                    bytes: b"a,b,c".to_vec(),
                    filename: "report.csv".to_string(),
                },
                DownloadedFile {
                    bytes: b"%PDF".to_vec(),
                    filename: "../../etc/passwd".to_string(),
                },
            ],
        )
        .unwrap();

        assert_eq!(prepared.paths().len(), 2);
        assert_eq!(prepared.paths()[0].file_name().unwrap(), "1-report.csv");
        assert!(prepared.paths()[1]
            .file_name()
            .unwrap()
            .to_string_lossy()
            .ends_with("passwd"));
        assert!(prepared
            .paths()
            .iter()
            .all(|path| path.starts_with(&prepared.directory)));
        for path in prepared.paths() {
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let directory = prepared.directory.clone();
        drop(prepared);
        assert!(!directory.exists());
        let _ = std::fs::remove_dir(cache);
    }

    #[test]
    fn rejects_excess_and_oversized_files() {
        let cache = temp_dir("rejected-files");
        assert!(PreparedFiles::create(
            &cache,
            (0..=MAX_FILE_COUNT)
                .map(|index| DownloadedFile {
                    bytes: b"x".to_vec(),
                    filename: format!("f{index}.bin"),
                })
                .collect(),
        )
        .is_err());
        assert!(PreparedFiles::create(
            &cache,
            vec![DownloadedFile {
                bytes: vec![0; MAX_FILE_BYTES + 1],
                filename: "big.bin".to_string(),
            }],
        )
        .is_err());
        let _ = std::fs::remove_dir(cache);
    }
}
