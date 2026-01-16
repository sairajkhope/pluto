.. _onshape_to_robot:

Onshape to Robot Descriptions
==============================

This is a command-line tool that allows you to convert Onshape robots to URDF and MJCF files for simulation and deployment.
The tool is built on top of the `onshape_to_robot <https://github.com/Rhoban/onshape-to-robot>`__ package.

Obtain the API key and secret key from the `Onshape developer portal <https://dev-portal.onshape.com/keys>`__.

We recommend storing your API key and secret in environment variables. You have two options:

**Option 1: Using a .env file (Recommended)**

Create a `.env` file in the project root (you can copy from `.env.example`):

::

   ONSHAPE_API=https://cad.onshape.com
   ONSHAPE_ACCESS_KEY=Your_Access_Key
   ONSHAPE_SECRET_KEY=Your_Secret_Key

**Option 2: Using .bashrc**

Add the following to your `.bashrc`:

::

   export ONSHAPE_API=https://cad.onshape.com
   export ONSHAPE_ACCESS_KEY=Your_Access_Key
   export ONSHAPE_SECRET_KEY=Your_Secret_Key


Read the `config doc <https://onshape-to-robot.readthedocs.io/en/latest/config.html>`__ first if you have any issues.

Run the following script and follow the instructions:

::

   python toddlerbot/descriptions/onshape_to_robot.py --robot <robot_name>


Please carefully read the terminal messages printed from this script.