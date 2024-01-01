import argparse
import os, sys, subprocess
import toml
import shutil

project_conf = toml.load('./pyproject.toml')
project_name = project_conf.get('project', {}).get('name')
project_version = project_conf.get('project', {}).get('version')
project_root = os.path.abspath('.')

parser = argparse.ArgumentParser(
    prog='Decompy Run',
    description='This is an utility script to perform various deployment mechanism like `npm run`'
)
parser.add_argument('action', choices=['build', 'document', 'deploy', 'test'])
args = parser.parse_args()

if args.action == 'build':
    x = input(f"Found {project_name} version {project_version} in pyproject.toml. Is that the latest version? [y/n]: ")
    if x == "y" or x == 'Y':
        print(f"Building {project_name} for version {project_version}")
        subprocess.Popen('python3 -m build', cwd = project_root, shell = True).wait()
elif args.action == 'document':
    print(f"Creating documentation for {project_name} for version {project_version}")
    for fname in ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md']:
        shutil.copy(fname, os.path.join('docs', fname))
    subprocess.Popen('make clean', cwd = os.path.join(project_root, 'docs'), shell = True).wait()
    subprocess.Popen('make html', cwd = os.path.join(project_root, 'docs'), shell = True).wait()

    # server the server
    subprocess.Popen('python -m http.server', cwd = os.path.join(project_root, 'docs', 'build', 'html'), 
                     shell = True).wait()
elif args.action == 'test':
    print(f"No test found")
else:
    raise ValueError("Invalid action")
