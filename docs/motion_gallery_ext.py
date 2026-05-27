"""
Sphinx extension to automatically generate the motion gallery from motion files.

This extension scans the motion/ directory for .lz4 files and generates
the gallery data (motions.json) during docs build. Users don't need to
manually maintain a motions.json file.
"""

import json
import os
import re
from pathlib import Path

import joblib


# Motions to exclude from the gallery
EXCLUDED_MOTIONS = {
    "arm_pose_dataset",
    "walk_zmp_2xc",
    "walk_zmp_2xm",
}


def get_motion_category(name):
    """Infer category from motion name."""
    name_lower = name.lower()

    # Locomotion motions (crawl, walk, climb, stairs - all movement)
    if any(kw in name_lower for kw in ['crawl', 'walk', 'stairs', 'climb']):
        return 'locomotion'

    # Acrobatics
    if any(kw in name_lower for kw in ['cartwheel', 'flip', 'jump']):
        return 'acrobatics'

    # Recovery / Getting up (get_up, get_down, get_off, roll to recover)
    if any(kw in name_lower for kw in ['get_up', 'get_down', 'get_off']):
        return 'recovery'

    # Manipulation (hold, grasp, pull_up bar, push_up exercise)
    if any(kw in name_lower for kw in ['hold', 'grasp', 'pull_up', 'push_up']):
        return 'manipulation'

    # Poses
    if any(kw in name_lower for kw in ['kneel', 'cuddle', 'pose', 'stand']):
        return 'poses'

    # Testing
    if 'test' in name_lower:
        return 'testing'

    return 'other'


def get_robot_from_name(name):
    """Extract robot type from motion name."""
    if '_2xm' in name:
        return 'toddlerbot_2xm'
    elif '_2xc' in name:
        return 'toddlerbot_2xc'
    else:
        # Default to 2xc
        return 'toddlerbot_2xc'


def format_display_name(name):
    """Convert snake_case name to display name."""
    # Extract robot suffix
    robot_suffix = ""
    if name.endswith('_2xc'):
        robot_suffix = " (2XC)"
    elif name.endswith('_2xm'):
        robot_suffix = " (2XM)"

    # Remove robot suffix from base name
    display = re.sub(r'_2x[cm]$', '', name)
    # Convert underscores to spaces and title case
    display = display.replace('_', ' ').title()

    return display + robot_suffix


def load_motion_metadata(motion_path):
    """Load motion file and extract metadata."""
    try:
        data = joblib.load(motion_path)

        # Get keyframes count (from actual keyframes list)
        num_keyframes = 0
        if 'keyframes' in data and data['keyframes']:
            num_keyframes = len(data['keyframes'])

        # Get trajectory length for duration calculation
        traj_length = 0
        if 'qpos' in data:
            traj_length = len(data['qpos'])
        elif num_keyframes > 0:
            traj_length = num_keyframes

        # Calculate duration (assuming 50Hz control rate)
        duration_seconds = traj_length / 50.0 if traj_length > 0 else 0.0

        return {
            'num_keyframes': num_keyframes,
            'duration_seconds': round(duration_seconds, 1),
        }
    except Exception as e:
        print(f"Warning: Could not load {motion_path}: {e}")
        return {
            'num_keyframes': 0,
            'duration_seconds': 0.0,
        }


def generate_motions_json(app):
    """Generate motions.json from motion directory during Sphinx build."""
    # Get paths
    docs_dir = Path(app.srcdir)
    project_root = docs_dir.parent
    motion_dir = project_root / 'motion'
    output_dir = docs_dir / '_static' / 'motion_gallery'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Scan motion files
    motion_files = sorted(motion_dir.glob('*.lz4'))

    # Skip certain files (datasets, etc.)
    skip_patterns = ['dataset', 'backup', 'old_']

    motions = []
    categories = set()
    robots = set()

    for motion_path in motion_files:
        name = motion_path.stem

        # Skip excluded motions
        if name in EXCLUDED_MOTIONS:
            continue

        # Skip files matching skip patterns
        if any(pattern in name.lower() for pattern in skip_patterns):
            continue

        # Get metadata
        metadata = load_motion_metadata(motion_path)

        # Get robot and category
        robot = get_robot_from_name(name)
        category = get_motion_category(name)

        categories.add(category)
        robots.add(robot)

        motion_entry = {
            'name': name,
            'display_name': format_display_name(name),
            'robot': robot,
            'category': category,
            'num_keyframes': metadata['num_keyframes'],
            'duration_seconds': metadata['duration_seconds'],
            'file_path': f'motion/{name}.lz4',
            'video_path': f'videos/{name}.mp4',
            'thumbnail_path': f'thumbnails/{name}.jpg',
            'download_url': f'https://github.com/hshi74/toddlerbot/raw/main/motion/{name}.lz4',
        }

        motions.append(motion_entry)

    # Sort motions by display name
    motions.sort(key=lambda m: m['display_name'])

    # Create the JSON structure
    gallery_data = {
        'total_motions': len(motions),
        'categories': sorted(list(categories)),
        'robots': sorted(list(robots)),
        'motions': motions,
    }

    # Write motions.json
    output_path = output_dir / 'motions.json'
    with open(output_path, 'w') as f:
        json.dump(gallery_data, f, indent=2)

    print(f"Motion Gallery: Generated {len(motions)} motions from {motion_dir}")


def setup(app):
    """Setup the Sphinx extension."""
    # Generate motions.json at build start
    app.connect('builder-inited', generate_motions_json)

    return {
        'version': '1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
