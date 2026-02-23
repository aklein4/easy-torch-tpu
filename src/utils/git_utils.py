
import git

import utils.constants as constants


def get_current_commit_hash():
    repo = git.Repo(
        path=constants.BASE_PATH,
        search_parent_directories=True
    )
    return repo.head.object.hexsha
