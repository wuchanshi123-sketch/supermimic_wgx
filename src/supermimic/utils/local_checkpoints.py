"""Expose the newest complete checkpoint at stable paths in the run directory."""

from pathlib import Path
from tempfile import TemporaryDirectory

_LATEST = Path(".latest")


def update_latest_checkpoint(checkpoint_dir: Path) -> None:
  """Point the run's weight and config links at a completed checkpoint."""
  run_dir = checkpoint_dir.parent
  for file in (*checkpoint_dir.glob("*.pt"), checkpoint_dir / "runner_config.json"):
    target = _LATEST / file.name
    link = run_dir / file.name
    if link.is_symlink() and link.readlink() == target:
      continue
    link.symlink_to(target)

  # All established file links switch versions together when this link is replaced.
  with TemporaryDirectory(dir=run_dir, prefix=".latest-") as temp_dir:
    pointer = Path(temp_dir) / _LATEST
    pointer.symlink_to(checkpoint_dir.name, target_is_directory=True)
    pointer.replace(run_dir / _LATEST)


def resolve_checkpoint_actor(actor_path: Path) -> Path:
  """Pin a latest alias to one snapshot before playback loads its sibling files."""
  if actor_path.is_symlink() and actor_path.readlink() == _LATEST / actor_path.name:
    return actor_path.resolve()
  return actor_path
