
import git

import utils.constants as constants


def get_current_commit_hash() -> str:
    """
    Get the hash of the repo's current git commit.

    Returns:
        str: The hash of the current git commit.
    """
    repo = git.Repo(
        path=constants.BASE_PATH,
        search_parent_directories=True
    )
    return repo.head.object.hexsha
