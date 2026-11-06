"""Load the native EndoChat teacher from a user-supplied installation."""

import sys
from pathlib import Path


def load_teacher(model_path, code_root=None, with_visual=True):
    if code_root is not None:
        root = Path(code_root).expanduser().resolve()
        if not (root / "SPHINX" / "sphinx.py").is_file():
            raise FileNotFoundError("teacher code root must contain SPHINX/sphinx.py")
        sys.path.insert(0, str(root))
    from SPHINX.sphinx import SPHINXModel

    teacher = (
        SPHINXModel.from_pretrained(pretrained_path=str(model_path), with_visual=with_visual)
        .to("cuda")
        .eval()
    )
    teacher.requires_grad_(False)
    return teacher
