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
    subprocess.Popen('sphinx-apidoc -o docsource/ -d 3 ./src', cwd = os.path.join(project_root), shell= True).wait()
    for fname in ['README.md', 'CHANGELOG.md', 'CONTRIBUTING.md', 'LICENSE']:
        shutil.copy(fname, os.path.join('docsource', fname))
    subprocess.Popen('make clean', cwd = os.path.join(project_root, 'docsource'), shell = True).wait()
    subprocess.Popen('make html', cwd = os.path.join(project_root, 'docsource'), shell = True).wait()

    # copy all the files from docsource to docs
    if os.path.exists(os.path.join(project_root, 'docs')) and os.path.isdir(os.path.join(project_root, 'docs')):
        shutil.rmtree(os.path.join(project_root, 'docs'))
    # copy html files
    shutil.copytree(os.path.join(project_root, 'docsource', '_build', 'html'), os.path.join(project_root, 'docs'))

    # server the server
    subprocess.Popen('python -m http.server', cwd = os.path.join(project_root, 'docs'), shell = True).wait()


elif args.action == 'test':
    print(f"No test found")
else:
    raise ValueError("Invalid action")
