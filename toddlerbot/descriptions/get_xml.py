"""OnShape robot assembly XML generation utility.

Downloads and processes robot assemblies from OnShape CAD platform, converting them
to MuJoCo XML format for simulation. Handles configuration generation, mesh processing,
and file cleanup.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from toddlerbot.utils.io_utils import pretty_write_xml


def load_env_local():
    """Load environment variables from .env.local file if it exists."""
    env_local_path = Path(".env.local")
    if env_local_path.exists():
        with open(env_local_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE format
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if key and key not in os.environ:
                        os.environ[key] = value


def process_xml_and_stl_files(assembly_path: str):
    """Processes XML and STL files within a specified assembly directory.

    Args:
        assembly_path (str): The path to the directory containing the XML and STL files.

    Raises:
        ValueError: If no XML file is found in the specified directory.
    """
    xml_path = os.path.join(assembly_path, "robot.xml")
    if not os.path.exists(xml_path):
        raise ValueError("No XML file found in the robot directory.")

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    pretty_write_xml(root, xml_path)

    # Delete PART and unmerged STL files
    assets_path = os.path.join(assembly_path, "assets")
    for entry in os.scandir(assets_path):
        if entry.is_file():
            os.remove(entry.path)
        elif entry.is_dir() and entry.name != "merged":
            shutil.rmtree(entry.path)


def run_onshape_to_robot(
    doc_id_list: List[str],
    assembly_list: List[str],
    element_id_list: List[str] = None,
    max_stl_size: int = 1
):
    """Downloads and converts OnShape assemblies to robot XML format.

    Args:
        doc_id_list: List of OnShape document IDs
        assembly_list: List of assembly names to process
        element_id_list: Optional list of OnShape element IDs (for direct element access)
        max_stl_size: Maximum STL file size in MB
    """
    assembly_dir = os.path.join("toddlerbot", "descriptions", "assemblies")

    if element_id_list is None:
        element_id_list = [None] * len(doc_id_list)

    # Process each assembly in series
    for doc_id, assembly_name, element_id in zip(doc_id_list, assembly_list, element_id_list):
        assembly_path = os.path.join(assembly_dir, assembly_name)

        if os.path.exists(assembly_path):
            shutil.rmtree(assembly_path)

        os.makedirs(assembly_path)
        json_file_path = os.path.join(assembly_path, "config.json")

        joint_properties_dict = {}
        equalities_dict = {}
        workspace_id = None  # Optional: specify workspace, otherwise uses default
        # element_id is passed as parameter, but can be overridden below if needed
        
        if "leg" in assembly_name:
            base_assembly_name: str = "leg"
            config_name = assembly_name.replace("leg_", "")
        elif "arm" in assembly_name:
            base_assembly_name = "arm"
            config_name = assembly_name.replace("arm_", "")
            for passive_joint in ["gripper_pinion", "gripper_pinion_mirror"]:
                joint_properties_dict[passive_joint] = {"actuated": False}
        elif assembly_name.startswith("sysID"):
            # For sysID robots: each assembly is standalone (no configuration variants)
            base_assembly_name = assembly_name
            config_name = None  # No configuration parameter needed
            
            # sysID robots share the same document and workspace
            workspace_id = "04f325bf94af0839fb12fa93"  # Common workspace for sysID assemblies
            # element_id: To be filled in per assembly (left as None for now, lookup by name)
        else:
            base_assembly_name = "toddlerbot"
            config_name = assembly_name
            for passive_joint in [
                "neck_pitch_front",
                "neck_pitch_back",
                "neck_pitch",
                "waist_roll",
                "waist_yaw",
            ]:
                joint_properties_dict[passive_joint] = {"actuated": False}

            equalities_dict["closing_neck_pitch*"] = {
                "solref": "0.004 1",
                "solimp": "0.9999 0.9999 0.001 0.5 2",
            }

        # Build json_data with conditional fields
        json_data = {
            "document_id": doc_id,
            "output_format": "mujoco",
            "robot_name": assembly_name,
            "assembly_name": base_assembly_name,
            "include_configuration_suffix": False,
            "draw_frames": True,
            "merge_stls": True,
            "simplify_stls": True,
            "maxSTLSize": max_stl_size,
            "joint_properties": joint_properties_dict,
            "equalities": equalities_dict,
        }
        
        # Add optional fields only if specified
        if workspace_id is not None:
            json_data["workspace_id"] = workspace_id
        if element_id is not None:
            json_data["element_id"] = element_id
        if config_name is not None:
            json_data["configuration"] = f"Configuration={config_name}"

        # Write the JSON data to a file
        with open(json_file_path, "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        # Execute the command
        # Try to find onshape-to-robot in conda/pip installation
        onshape_cmd = shutil.which("onshape-to-robot")
        if not onshape_cmd:
            # Try in the same directory as python
            python_dir = os.path.dirname(sys.executable)
            potential_path = os.path.join(python_dir, "onshape-to-robot")
            if os.path.exists(potential_path):
                onshape_cmd = potential_path
        
        if onshape_cmd:
            subprocess.run([onshape_cmd, assembly_path])
        else:
            print(f"⚠️  onshape-to-robot command not found. Skipping {assembly_name}")
            print("   Make sure onshape-to-robot is installed: pip install onshape-to-robot")
            continue

        process_xml_and_stl_files(assembly_path)


def main():
    """Main entry point for OnShape assembly processing."""
    # Load environment variables from .env.local before processing
    load_env_local()
    
    parser = argparse.ArgumentParser(description="Process the xml.")
    parser.add_argument(
        "--doc-id-list",
        type=str,
        nargs="+",  # Indicates that one or more arguments will be consumed.
        required=True,
        help="The names of the documents. Need to match the names in OnShape.",
    )
    parser.add_argument(
        "--assembly-list",
        type=str,
        nargs="+",  # Indicates that one or more arguments will be consumed.
        required=True,
        help="The names of the assemblies. Need to match the names in OnShape.",
    )
    parser.add_argument(
        "--element-id-list",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of element IDs for direct element access.",
    )
    args = parser.parse_args()

    run_onshape_to_robot(args.doc_id_list, args.assembly_list, args.element_id_list)


if __name__ == "__main__":
    main()
