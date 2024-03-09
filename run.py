import argparse
import os, sys, subprocess
import toml
import shutil

project_conf = toml.load('./pyproject.toml')
project_name = project_conf.get('project', {}).get('name')
project_root = os.path.abspath('.')
python_cmd = "python"

parser = argparse.ArgumentParser(
    prog='Decompy Run',
    description='This is an utility script to perform various deployment mechanism like `npm run`'
)
parser.add_argument('action', choices=['build', 'document', 'deploy', 'test'])
args = parser.parse_args()


def build_action():
    print(f"Building {project_name}")
    #remove existing dist folder
    if os.path.exists(os.path.join(project_root, 'dist')) and os.path.isdir(os.path.join(project_root, 'dist')):
        shutil.rmtree(os.path.join(project_root, 'dist'))
    os.mkdir(os.path.join(project_root, 'dist'))
    subprocess.Popen(f"{python_cmd} -m build", cwd = project_root, shell = True).wait()

def document_action():
    print(f"Creating documentation for {project_name}")
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
    subprocess.Popen(f"{python_cmd} -m http.server", cwd = os.path.join(project_root, 'docs'), shell = True).wait()

def deploy_action():
    print(f"Deploying to PyPI")
    subprocess.Popen('twine check dist/*', cwd = project_root, shell = True).wait()
    subprocess.Popen('twine upload dist/* --verbose', cwd = project_root, shell = True).wait()
    
def test_action():
    print(f"Starting test")
    subprocess.Popen("pytest --cov-report term-missing --cov=decompy tests/", cwd=project_root, shell = True).wait()


if args.action == 'build':
    build_action()
elif args.action == 'document':
    document_action()
elif args.action == 'test':
    test_action()
elif args.action == 'deploy':
    build_action()
    deploy_action()
else:
    raise ValueError("Invalid action")
