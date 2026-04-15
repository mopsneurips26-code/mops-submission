"""This script sets up a private macros file.
The private macros file (macros_private.py) is not tracked by git,
allowing user-specific settings that are not tracked by git.
This script checks if macros_private.py exists.
If applicable, it creates the private macros at robocasa/macros_private.py.
"""

import os
import shutil

import robocasa

if __name__ == "__main__":
    base_path = robocasa.__path__[0]
    macros_path = os.path.join(base_path, "macros.py")
    macros_private_path = os.path.join(base_path, "macros_private.py")

    print("Setting up private macros file...")

    if not os.path.exists(macros_path):
        print(f"{macros_path} does not exist! Aborting...")

    if os.path.exists(macros_private_path):
        ans = input(f"{macros_private_path} already exists! \noverwrite? (y/n)\n")

        if ans == "y":
            print("REMOVING")
        else:
            exit()

    shutil.copyfile(macros_path, macros_private_path)
    print(f"copied {macros_path}\nto {macros_private_path}")
