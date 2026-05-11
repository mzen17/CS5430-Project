import subprocess


scripts = [
    'vae.py',
    'lem.py',
    'latent.py',
    'lla.py',
]


def main():
    for script in scripts:
        print(f"running {script}")
        subprocess.run(['uv', 'run', 'python', script], check=True)


if __name__ == '__main__':
    main()
