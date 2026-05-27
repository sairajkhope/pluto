.. _motion_gallery:

Motion Gallery
==============

The Motion Gallery is a collection of reference motions for ToddlerBot. These motions can be used for:

- **Imitation learning**: Train policies to reproduce these motions
- **Motion primitives**: Chain together basic movements for complex behaviors
- **Testing and debugging**: Verify robot hardware and software with known-good trajectories

.. raw:: html

   <style>
   /* Light mode colors (default) */
   .motion-gallery-container {
       --gallery-bg: #f5f5f5;
       --gallery-header-bg: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
       --gallery-controls-bg: #e2e8f0;
       --gallery-card-bg: #ffffff;
       --gallery-thumbnail-bg: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
       --gallery-title-color: #1a202c;
       --gallery-text-color: #4a5568;
       --gallery-badge-bg: #e2e8f0;
       --gallery-badge-color: #4a5568;
       --gallery-input-bg: #ffffff;
       --gallery-input-border: #cbd5e0;
       --gallery-input-color: #1a202c;
       margin: 20px 0;
       max-width: 1400px;
   }
   /* Dark mode - when body has data-theme="dark" */
   body[data-theme="dark"] .motion-gallery-container {
       --gallery-bg: #1a1a2e;
       --gallery-header-bg: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
       --gallery-controls-bg: #252545;
       --gallery-card-bg: #1e1e3f;
       --gallery-thumbnail-bg: linear-gradient(135deg, #2d2d5a 0%, #1a1a3e 100%);
       --gallery-title-color: #ffffff;
       --gallery-text-color: #888888;
       --gallery-badge-bg: #3a3a6a;
       --gallery-badge-color: #aaaaaa;
       --gallery-input-bg: #1a1a3a;
       --gallery-input-border: #3a3a6a;
       --gallery-input-color: #ffffff;
   }
   /* Auto mode - respect system preference when no explicit theme set */
   @media (prefers-color-scheme: dark) {
       body:not([data-theme="light"]) .motion-gallery-container {
           --gallery-bg: #1a1a2e;
           --gallery-header-bg: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
           --gallery-controls-bg: #252545;
           --gallery-card-bg: #1e1e3f;
           --gallery-thumbnail-bg: linear-gradient(135deg, #2d2d5a 0%, #1a1a3e 100%);
           --gallery-title-color: #ffffff;
           --gallery-text-color: #888888;
           --gallery-badge-bg: #3a3a6a;
           --gallery-badge-color: #aaaaaa;
           --gallery-input-bg: #1a1a3a;
           --gallery-input-border: #3a3a6a;
           --gallery-input-color: #ffffff;
       }
   }
   .motion-gallery-container .header {
       background: var(--gallery-header-bg);
       padding: 20px;
       border-radius: 8px;
       margin-bottom: 20px;
       color: #fff;
   }
   .motion-gallery-container .logo {
       font-size: 1.5rem;
       margin: 0;
   }
   .motion-gallery-container .subtitle {
       margin: 5px 0 0 0;
       opacity: 0.8;
   }
   .motion-gallery-container .footer {
       display: none;
   }
   .motion-gallery-container .controls-bar {
       background: var(--gallery-controls-bg);
       padding: 15px;
       border-radius: 8px;
       margin-bottom: 20px;
   }
   .motion-gallery-container .motion-grid {
       display: grid;
       grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
       gap: 16px;
   }
   .motion-gallery-container .motion-card {
       background: var(--gallery-card-bg);
       border-radius: 8px;
       overflow: hidden;
       cursor: pointer;
       transition: transform 0.2s, box-shadow 0.2s;
   }
   .motion-gallery-container .motion-card:hover {
       transform: translateY(-4px);
       box-shadow: 0 8px 25px rgba(0,0,0,0.15);
   }
   .motion-gallery-container .motion-thumbnail {
       height: 160px;
       background: var(--gallery-thumbnail-bg);
       display: flex;
       align-items: center;
       justify-content: center;
       color: var(--gallery-text-color);
       position: relative;
   }
   .motion-gallery-container .motion-thumbnail img {
       width: 100%;
       height: 100%;
       object-fit: cover;
   }
   .motion-gallery-container .motion-info-card {
       padding: 15px;
   }
   .motion-gallery-container .motion-title {
       font-size: 1rem;
       margin: 0 0 8px 0;
       color: var(--gallery-title-color);
   }
   .motion-gallery-container .motion-meta {
       display: flex;
       gap: 8px;
       margin-bottom: 8px;
   }
   .motion-gallery-container .meta-badge {
       font-size: 0.75rem;
       padding: 2px 8px;
       border-radius: 4px;
       background: var(--gallery-badge-bg);
       color: var(--gallery-badge-color);
   }
   .motion-gallery-container .motion-stats {
       font-size: 0.8rem;
       color: var(--gallery-text-color);
   }
   .motion-gallery-container .search-input,
   .motion-gallery-container .filter-select {
       background: var(--gallery-input-bg);
       border: 1px solid var(--gallery-input-border);
       color: var(--gallery-input-color);
       padding: 8px 12px;
       border-radius: 4px;
   }
   .motion-gallery-container .filter-group label {
       color: var(--gallery-text-color);
   }
   .motion-gallery-container .stats {
       color: var(--gallery-text-color);
   }
   .motion-gallery-container .modal {
       display: none;
       position: fixed;
       top: 0;
       left: 0;
       width: 100%;
       height: 100%;
       background: rgba(0,0,0,0.8);
       z-index: 9999;
       align-items: center;
       justify-content: center;
   }
   .motion-gallery-container .modal.active {
       display: flex;
   }
   .motion-gallery-container .modal-content {
       background: var(--gallery-card-bg);
       border-radius: 12px;
       max-width: 800px;
       width: 90%;
       max-height: 90vh;
       overflow-y: auto;
   }
   .motion-gallery-container .modal-content h2 {
       color: var(--gallery-title-color);
   }
   .motion-gallery-container .modal-close {
       position: absolute;
       top: 15px;
       right: 20px;
       font-size: 2rem;
       color: #fff;
       cursor: pointer;
   }
   .motion-gallery-container .controls-grid {
       display: flex;
       flex-wrap: wrap;
       gap: 15px;
       align-items: center;
   }
   .motion-gallery-container .filter-group {
       display: flex;
       align-items: center;
       gap: 8px;
   }
   .motion-gallery-container .stats-row {
       display: flex;
       justify-content: space-between;
       align-items: center;
       margin-top: 15px;
   }
   .motion-gallery-container .stats {
       color: #888;
       font-size: 0.9rem;
   }
   .motion-gallery-container .btn-create-new {
       display: none; /* Hide create button in docs */
   }
   .motion-gallery-container .duration-badge {
       position: absolute;
       bottom: 8px;
       right: 8px;
       background: rgba(0,0,0,0.7);
       color: #fff;
       padding: 2px 6px;
       border-radius: 4px;
       font-size: 0.75rem;
   }
   </style>

   <div class="motion-gallery-container">
       <!-- Header -->
       <header class="header">
           <h1 class="logo">ToddlerBot Motion Gallery</h1>
           <p class="subtitle">Reference motion dataset for humanoid robotics</p>
       </header>

       <!-- Search & Filter Bar -->
       <div class="controls-bar">
           <div class="controls-grid">
               <!-- Search -->
               <div class="search-box">
                   <input
                       type="text"
                       id="searchInput"
                       placeholder="Search motions..."
                       class="search-input"
                   >
               </div>

               <!-- Category Filter (populated dynamically) -->
               <div class="filter-group">
                   <label for="categoryFilter">Category:</label>
                   <select id="categoryFilter" class="filter-select">
                       <option value="all">All Categories</option>
                   </select>
               </div>

               <!-- Robot Filter (populated dynamically) -->
               <div class="filter-group">
                   <label for="robotFilter">Robot:</label>
                   <select id="robotFilter" class="filter-select">
                       <option value="all">All Robots</option>
                   </select>
               </div>

               <!-- Sort -->
               <div class="filter-group">
                   <label for="sortBy">Sort:</label>
                   <select id="sortBy" class="filter-select">
                       <option value="name">Name</option>
                       <option value="duration">Duration</option>
                       <option value="category">Category</option>
                   </select>
               </div>
           </div>

           <!-- Stats -->
           <div class="stats-row">
               <div class="stats">
                   <span id="statsText">Loading motions...</span>
               </div>
           </div>
       </div>

       <!-- Motion Grid -->
       <div id="motionGrid" class="motion-grid">
           <!-- Motion cards will be inserted here by JavaScript -->
       </div>

       <!-- Empty State -->
       <div id="emptyState" class="empty-state" style="display: none; text-align: center; padding: 40px; color: #888;">
           <p>No motions found matching your filters.</p>
       </div>

       <!-- Video Modal -->
       <div id="videoModal" class="modal">
           <div class="modal-content">
               <span class="modal-close">&times;</span>
               <div class="modal-header" style="padding: 20px;">
                   <h2 id="modalTitle" style="margin: 0; color: #fff;">Motion Title</h2>
               </div>
               <div class="modal-body" style="padding: 0 20px 20px;">
                   <!-- Video Player -->
                   <div class="video-container" style="background: #000; border-radius: 8px; overflow: hidden;">
                       <video id="modalVideo" controls autoplay loop style="width: 100%; display: block;">
                           <source id="modalVideoSource" src="" type="video/mp4">
                           Your browser does not support video playback.
                       </video>
                   </div>

                   <!-- Motion Info -->
                   <div class="motion-info" style="margin-top: 20px;">
                       <div class="info-grid" style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                           <div class="info-item" style="text-align: center;">
                               <span class="info-label" style="display: block; color: #888; font-size: 0.8rem;">Duration</span>
                               <span class="info-value" id="modalDuration" style="color: #fff; font-weight: bold;">-</span>
                           </div>
                           <div class="info-item" style="text-align: center;">
                               <span class="info-label" style="display: block; color: #888; font-size: 0.8rem;">Keyframes</span>
                               <span class="info-value" id="modalKeyframes" style="color: #fff; font-weight: bold;">-</span>
                           </div>
                           <div class="info-item" style="text-align: center;">
                               <span class="info-label" style="display: block; color: #888; font-size: 0.8rem;">Robot</span>
                               <span class="info-value" id="modalRobot" style="color: #fff; font-weight: bold;">-</span>
                           </div>
                           <div class="info-item" style="text-align: center;">
                               <span class="info-label" style="display: block; color: #888; font-size: 0.8rem;">Category</span>
                               <span class="info-value" id="modalCategory" style="color: #fff; font-weight: bold;">-</span>
                           </div>
                       </div>

                       <!-- Action Buttons -->
                       <div class="modal-actions" style="margin-top: 20px; display: flex; gap: 15px; justify-content: center;">
                           <a id="modalDownload" href="#" download class="btn-download" style="background: #4CAF50; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none;">
                               Download Motion File (.lz4)
                           </a>
                       </div>
                   </div>
               </div>
           </div>
       </div>

       <!-- Hidden create modal (required by JS but not shown) -->
       <div id="createModal" class="modal" style="display: none;"></div>
   </div>

   <script src="../_static/motion_gallery/gallery.js"></script>


Using Motions
-------------

To load and use motions in your code:

.. code-block:: python

   import joblib

   # Load motion data
   motion_data = joblib.load("motion/crawl_2xc.lz4")

   # Access qpos trajectory
   qpos_trajectory = motion_data["qpos"]  # Shape: (T, 51)

   # Access action trajectory (if available)
   if "action" in motion_data:
       action_trajectory = motion_data["action"]  # Shape: (T, 30)


Motion File Format (.lz4)
-------------------------

Each ``.lz4`` file is a compressed dictionary containing both sparse keyframes and dense interpolated trajectories.

Dictionary Structure
~~~~~~~~~~~~~~~~~~~~

**Required keys:**

.. code-block:: python

   {
       # Time vector (dense 50Hz)
       'time': [0.0, 0.02, 0.04, ...],        # Shape: (T,)

       # Full joint state trajectory
       'qpos': [[...], [...], ...],           # Shape: (T, 51)

       # Action trajectory (None if generated from keyframe interpolation)
       'action': [[...], [...], ...],         # Shape: (T, 30) or None

       # Keyframe data (user-defined sparse poses)
       'keyframes': [
           {
               'name': 'cartwheel_000',
               'motor_pos': [...],            # Motor positions, shape: (30,)
               'joint_pos': [...],            # Joint angles, shape: (30,)
               'qpos': [...]                  # Full state, shape: (51,)
           },
           ...
       ],

       # Timing sequence (keyframe name, cumulative time in seconds)
       'timed_sequence': [
           ('cartwheel_000', 0.5),
           ('cartwheel_001', 0.8),
           ...
       ],
   }

**Optional keys (new format with physics data):**

.. code-block:: python

   {
       # Body poses from MuJoCo simulation
       'body_pos': [...],                     # Shape: (T, num_bodies, 3)
       'body_quat': [...],                    # Shape: (T, num_bodies, 4)

       # Site poses
       'site_pos': [...],                     # Shape: (T, num_sites, 3)
       'site_quat': [...],                    # Shape: (T, num_sites, 4)

       # Velocity data
       'body_lin_vel': [...],                 # Shape: (T, num_bodies, 3)
       'body_ang_vel': [...],                 # Shape: (T, num_bodies, 3)
       'motor_vel': [...],                    # Shape: (T, 30)
       'joint_vel': [...],                    # Shape: (T, 30)

       # Frame type flag
       'is_robot_relative_frame': False,
   }

**Old format keys (for backward compatibility):**

.. code-block:: python

   {
       'body_pose': [...],                    # Concatenated pos/quat arrays
       'site_pose': [...],
   }

qpos Format (51 values)
~~~~~~~~~~~~~~~~~~~~~~~

The ``qpos`` array contains 7 base pose values + 44 joint angles:

.. code-block:: text

   qpos[0:7]   - Base pose (free joint)
       [0:3]   x, y, z position (meters)
       [3:7]   qw, qx, qy, qz orientation (quaternion)

   qpos[7:51]  - Joint angles (44 values)
       [7]     neck_yaw_drive
       [8]     neck_yaw_driven
       [9]     neck_pitch
       [10]    neck_pitch_act
       [11]    neck_pitch_front    # passive
       [12]    neck_pitch_back     # passive
       [13]    waist_yaw
       [14]    waist_roll
       [15]    waist_act_1
       [16]    waist_act_2
       [17]    left_hip_pitch
       [18]    left_hip_roll
       [19]    left_hip_yaw_driven
       [20]    left_hip_yaw_drive
       [21]    left_knee
       [22]    left_ankle_pitch
       [23]    left_ankle_roll
       [24]    right_hip_pitch
       [25]    right_hip_roll
       [26]    right_hip_yaw_driven
       [27]    right_hip_yaw_drive
       [28]    right_knee
       [29]    right_ankle_pitch
       [30]    right_ankle_roll
       [31]    left_shoulder_pitch
       [32]    left_shoulder_roll
       [33]    left_shoulder_yaw_drive
       [34]    left_shoulder_yaw_driven
       [35]    left_elbow_roll
       [36]    left_elbow_yaw_drive
       [37]    left_elbow_yaw_driven
       [38]    left_wrist_pitch_driven
       [39]    left_wrist_pitch_drive
       [40]    left_wrist_roll
       [41]    right_shoulder_pitch
       [42]    right_shoulder_roll
       [43]    right_shoulder_yaw_drive
       [44]    right_shoulder_yaw_driven
       [45]    right_elbow_roll
       [46]    right_elbow_yaw_drive
       [47]    right_elbow_yaw_driven
       [48]    right_wrist_pitch_driven
       [49]    right_wrist_pitch_drive
       [50]    right_wrist_roll

Motors vs Joints
~~~~~~~~~~~~~~~~

ToddlerBot has **30 motors** but **44 MuJoCo joints** due to gear transmissions:

- **Spur gear joints**: Motors ending in ``_drive`` have corresponding ``_driven`` joints
- **Parallel linkage**: Motors ending in ``_act`` have ``_front`` and ``_back`` passive joints
- **Bevel gear**: ``waist_act_1`` and ``waist_act_2`` motors control ``waist_roll`` and ``waist_yaw`` joints

The keyframes store both representations:

- ``motor_pos``: 30 values (what you command)
- ``joint_pos``: 30 values (corresponding joint angles)
- ``qpos``: 51 values (full MuJoCo state including all passive/driven joints)

Creating New Motions
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m toddlerbot.tools.edit_keyframe_viser \
       --robot toddlerbot_2xc \
       --task my_motion_2xc \
       --run-name my_motion_2xc