import os
from pathlib import Path
import re


def pattern_search(
    dir_outer: str,
    pattern: str,
    max_depth: int = None,
    return_pattern: bool = False,
    natsorted: bool = True,
):
    """
    Search for files and/or folders in a directory that match a specific regular expression.

    Modified from find_paths function in bnpm

    Args:
        dir_outer (str):
            Path to directory to search
        pattern (str):
            Regular expression to match. For details, see https://docs.python.org/3/library/re.html
        max_depth (int):
            Maximum depth to recursive search. If None, search all subdirectories.
        return_pattern (bool):
            Whether to return the list of patterns found during searching.
        natsorted (bool):
            Whether to sort the output using natural sorting
             with the natsort package.

    Returns:
        tuple:
            paths (list):
                List of paths to files and/or folders that match the pattern.
            patterns (list):
                List of patterns that were matched.
    """

    def _get_paths_recursive_inner(dir_inner, max_depth, depth=0):
        paths = []
        patterns = []
        if max_depth is not None:
            for sub_path in os.listdir(dir_inner):
                path = os.path.join(dir_inner, sub_path)
                pattern_match = re.search(pattern, path)
                if pattern_match is not None:
                    paths.append(path)
                    patterns.append(pattern_match.group()) if return_pattern else None
                if depth < max_depth:
                    deeper_paths, deeper_patterns = _get_paths_recursive_inner(
                        path, max_depth, depth=depth + 1
                    )
                    paths += deeper_paths
                    patterns += deeper_patterns
        else:
            all_paths = Path(dir_inner).glob("**/*")
            for path in all_paths:
                pattern_match = re.search(pattern, str(path))
                if pattern_match is not None:
                    paths.append(str(path))
                    patterns.append(pattern_match.group()) if return_pattern else None
        return paths, patterns

    paths, patterns = _get_paths_recursive_inner(dir_outer, max_depth, depth=0)
    if natsorted:
        import natsort

        paths = natsort.natsorted(paths)
        patterns = natsort.natsorted(patterns) if return_pattern else None

    return paths, patterns
