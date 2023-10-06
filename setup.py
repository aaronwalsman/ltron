import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="ltron",
    version="0.0.6",
    author="Aaron Walsman",
    author_email="aaronwalsman@gmail.com",
    description='LEGO interactive machine learning environment.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaronwalsman/ltron",
    install_requires = [
        #'gym==0.17.3',
        #'gym==0.21.0',
        #'gym==0.26.0',
        'gymnasium>=0.26.3',
        'multiset',
        'numpy',
        'scipy',
        'pyquaternion',
        'gdown',
        'tqdm',
        'splendor-render',
        'webdataset',
    ],
    packages=setuptools.find_packages(),
    entry_points = {
        'console_scripts' : [
            'ltron_viewer=ltron.scripts.ltron_viewer:main',
            'ltron_asset_installer=ltron.scripts.ltron_asset_installer:main',
            'ltron_make_license=ltron.scripts.ltron_make_license:main',
            'ltron_make_symmetry_table=ltron.scripts.ltron_make_symmetry_table:'
                'main',
            'ltron_generate_episodes=ltron.dataset.break_and_make:'
                'generate_episodes_for_dataset',
            'ltron_clean_omr=ltron.dataset.omr_clean.ultimate_cleanup:'
                'clean_omr',
            'ltron_build_rc_dataset=ltron.dataset.rc:build_rc_dataset',
            'live_break_and_make=ltron.gym.envs.live_break_and_make:main',
            'ltron_generate_episode_collection='
                'ltron.dataset.generate_episode_collection:'
                'generate_episode_collection',
            'ltron_env_interface='
                'ltron.gym.ltron_env_interface:ltron_env_interface',
            'ltron_regenerate_class_labels='
                'ltron.dataset.class_labels:regenerate_class_labels',
        ],
        'gymnasium.envs' : [
            '__root__ = ltron.gym.register:register_ltron_envs',
        ]
    },
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]
)
