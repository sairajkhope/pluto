// Motion Gallery JavaScript
// Auto-generated from motion/ directory during docs build

let allMotions = [];
let filteredMotions = [];
let galleryData = {};

// Base path for assets (auto-detect based on script location)
const GALLERY_BASE_PATH = (function() {
    const scripts = document.getElementsByTagName('script');
    for (let i = 0; i < scripts.length; i++) {
        if (scripts[i].src.includes('gallery.js')) {
            return scripts[i].src.replace('gallery.js', '');
        }
    }
    return '_static/motion_gallery/';
})();

// Load motions data
async function loadMotions() {
    try {
        const response = await fetch(GALLERY_BASE_PATH + 'motions.json');
        galleryData = await response.json();
        allMotions = galleryData.motions || [];
        filteredMotions = allMotions;

        console.log(`Loaded ${allMotions.length} motions`);

        // Populate filter dropdowns dynamically
        populateFilters();

        renderMotions();
        updateStats();
    } catch (error) {
        console.error('Error loading motions:', error);
        document.getElementById('motionGrid').innerHTML = `
            <div style="grid-column: 1/-1; text-align: center; color: #ff4444; padding: 40px;">
                <p>Error loading motion gallery. Please make sure motions.json exists.</p>
            </div>
        `;
    }
}

// Populate filter dropdowns from loaded data
function populateFilters() {
    // Populate category filter
    const categoryFilter = document.getElementById('categoryFilter');
    if (categoryFilter && galleryData.categories) {
        categoryFilter.innerHTML = '<option value="all">All Categories</option>';
        galleryData.categories.forEach(cat => {
            const option = document.createElement('option');
            option.value = cat;
            option.textContent = cat.charAt(0).toUpperCase() + cat.slice(1);
            categoryFilter.appendChild(option);
        });
    }

    // Populate robot filter
    const robotFilter = document.getElementById('robotFilter');
    if (robotFilter && galleryData.robots) {
        robotFilter.innerHTML = '<option value="all">All Robots</option>';
        galleryData.robots.forEach(robot => {
            const option = document.createElement('option');
            option.value = robot;
            option.textContent = formatRobotName(robot);
            robotFilter.appendChild(option);
        });
    }
}

// Render motion cards
function renderMotions() {
    const grid = document.getElementById('motionGrid');
    const emptyState = document.getElementById('emptyState');
    
    if (filteredMotions.length === 0) {
        grid.style.display = 'none';
        emptyState.style.display = 'block';
        return;
    }
    
    grid.style.display = 'grid';
    emptyState.style.display = 'none';
    
    grid.innerHTML = filteredMotions.map(motion => createMotionCard(motion)).join('');
    
    // Add click listeners
    document.querySelectorAll('.motion-card').forEach((card, index) => {
        card.addEventListener('click', () => openModal(filteredMotions[index]));
    });
}

// Helper to resolve asset paths
function resolvePath(path) {
    if (!path) return '';
    if (path.startsWith('http')) return path;
    return GALLERY_BASE_PATH + path;
}

// Create motion card HTML
function createMotionCard(motion) {
    const hasVideo = motion.video_path && motion.video_path.includes('.mp4');
    const hasThumbnail = motion.thumbnail_path;
    const thumbnailPath = resolvePath(motion.thumbnail_path);

    return `
        <div class="motion-card" data-motion="${motion.name}">
            <div class="motion-thumbnail ${!hasVideo ? 'no-video' : ''}">
                ${hasThumbnail ? `<img src="${thumbnailPath}" alt="${motion.display_name}" onerror="this.style.display='none'">` : ''}
                ${motion.duration_seconds > 0 ? `<span class="duration-badge">${formatDuration(motion.duration_seconds)}</span>` : ''}
            </div>
            <div class="motion-info-card">
                <h3 class="motion-title">${motion.display_name}</h3>
                <div class="motion-meta">
                    <span class="meta-badge category">${motion.category}</span>
                    <span class="meta-badge robot">${formatRobotName(motion.robot)}</span>
                </div>
                <div class="motion-stats">
                    ${motion.num_keyframes} keyframes · ${motion.duration_seconds}s
                </div>
            </div>
        </div>
    `;
}

// Open video modal
function openModal(motion) {
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('modalVideo');
    const videoSource = document.getElementById('modalVideoSource');
    const videoContainer = video.parentElement;

    // Set video source (resolve path)
    const videoPath = resolvePath(motion.video_path);
    videoSource.src = videoPath;

    // Handle video load error - show placeholder
    video.onerror = () => {
        videoContainer.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:center;height:300px;background:#1a1a3a;color:#888;flex-direction:column;gap:10px;">
                <span style="font-size:48px;">🎬</span>
                <span>Video not available</span>
                <span style="font-size:12px;">Download the motion file to preview locally</span>
            </div>
        `;
    };

    video.load();

    // Set motion info
    document.getElementById('modalTitle').textContent = motion.display_name;
    document.getElementById('modalDuration').textContent = `${motion.duration_seconds}s`;
    document.getElementById('modalKeyframes').textContent = motion.num_keyframes;
    document.getElementById('modalRobot').textContent = formatRobotName(motion.robot);
    document.getElementById('modalCategory').textContent = motion.category;

    // For download URL, use GitHub raw URL for docs deployment
    const downloadUrl = motion.download_url.startsWith('http')
        ? motion.download_url
        : `https://github.com/hshi74/toddlerbot/raw/main/motion/${motion.name}.lz4`;
    document.getElementById('modalDownload').href = downloadUrl;
    document.getElementById('modalDownload').download = `${motion.name}.lz4`;

    // Setup edit button (hide if not running locally)
    const editBtn = document.getElementById('modalEdit');
    if (editBtn) {
        editBtn.onclick = (e) => {
            e.preventDefault();
            loadInViser(motion);
        };
    }

    // Show modal
    modal.classList.add('active');

    // Play video if available
    video.play().catch(err => {
        console.log('Video autoplay prevented:', err);
    });
}

// Load motion in Viser editor
function loadInViser(motion) {
    const loadingMsg = document.createElement('div');
    loadingMsg.style.cssText = 'position:fixed;top:20px;right:20px;background:#4CAF50;color:white;padding:15px 25px;border-radius:8px;z-index:10000;box-shadow:0 4px 6px rgba(0,0,0,0.3);font-family:system-ui;';
    loadingMsg.textContent = `Loading ${motion.display_name} in Viser...`;
    document.body.appendChild(loadingMsg);
    
    const apiUrl = `http://localhost:8082/reload?motion=${encodeURIComponent(motion.name)}&robot=${encodeURIComponent(motion.robot)}`;
    
    fetch(apiUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                loadingMsg.style.background = '#4CAF50';
                const endTime = Date.now() + 10000;
                let countdownInterval = null;
                let launched = false;

                const finishLaunch = () => {
                    if (launched) {
                        return;
                    }
                    launched = true;
                    if (countdownInterval !== null) {
                        clearInterval(countdownInterval);
                    }
                    loadingMsg.innerHTML = `
                        <div style="font-weight:bold;">✅ ${data.message}</div>
                        <div style="font-size:12px;margin-top:4px;">Opening Viser now...</div>
                    `;

                    const opened = window.open('http://localhost:8080', '_blank');
                    if (!opened) {
                        loadingMsg.style.background = '#ff9800';
                        loadingMsg.innerHTML = `
                            <div style="font-weight:bold;">⚠️ Popup Blocked</div>
                            <div style="font-size:12px;margin-top:8px;">
                                <a href="http://localhost:8080" target="_blank" style="color:white;text-decoration:underline;">
                                    Click here to open Viser
                                </a>
                            </div>
                        `;
                    } else {
                        closeModal();
                    }

                    setTimeout(() => loadingMsg.remove(), 2000);
                };

                const updateCountdown = () => {
                    const remainingMs = endTime - Date.now();
                    const remainingSeconds = Math.max(0, Math.ceil(remainingMs / 1000));
                    loadingMsg.innerHTML = `
                        <div style="font-weight:bold;">✅ ${data.message}</div>
                        <div style="font-size:12px;margin-top:4px;">Opening in ${remainingSeconds}...</div>
                    `;
                    if (remainingSeconds <= 0) {
                        finishLaunch();
                    }
                };

                updateCountdown();
                countdownInterval = setInterval(updateCountdown, 1000);
            } else {
                loadingMsg.style.background = '#f44336';
                loadingMsg.textContent = `❌ Error: ${data.message}`;
                setTimeout(() => loadingMsg.remove(), 5000);
            }
        })
        .catch(error => {
            loadingMsg.style.background = '#f44336';
            loadingMsg.textContent = `❌ Connection Error: ${error.message}`;
            setTimeout(() => loadingMsg.remove(), 5000);
        });
}

// Close modal
function closeModal() {
    const modal = document.getElementById('videoModal');
    const video = document.getElementById('modalVideo');
    
    modal.classList.remove('active');
    video.pause();
    video.currentTime = 0;
}

// Filter motions
function filterMotions() {
    const searchTerm = document.getElementById('searchInput').value.toLowerCase();
    const category = document.getElementById('categoryFilter').value;
    const robot = document.getElementById('robotFilter').value;
    const sortBy = document.getElementById('sortBy').value;
    
    // Filter
    filteredMotions = allMotions.filter(motion => {
        const matchesSearch = motion.display_name.toLowerCase().includes(searchTerm) || 
                              motion.name.toLowerCase().includes(searchTerm);
        const matchesCategory = category === 'all' || motion.category === category;
        const matchesRobot = robot === 'all' || motion.robot === robot;
        
        return matchesSearch && matchesCategory && matchesRobot;
    });
    
    // Sort
    filteredMotions.sort((a, b) => {
        if (sortBy === 'name') {
            return a.display_name.localeCompare(b.display_name);
        } else if (sortBy === 'duration') {
            return b.duration_seconds - a.duration_seconds;
        } else if (sortBy === 'category') {
            return a.category.localeCompare(b.category);
        }
        return 0;
    });
    
    renderMotions();
    updateStats();
}

// Update stats text
function updateStats() {
    const statsText = document.getElementById('statsText');
    const total = allMotions.length;
    const showing = filteredMotions.length;
    
    if (showing === total) {
        statsText.textContent = `Showing all ${total} motions`;
    } else {
        statsText.textContent = `Showing ${showing} of ${total} motions`;
    }
}

// Format duration (seconds to MM:SS)
function formatDuration(seconds) {
    if (seconds === 0) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Format robot name
function formatRobotName(robot) {
    return robot.replace('toddlerbot_', '').toUpperCase();
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Load motions
    loadMotions();
    
    // Search input
    document.getElementById('searchInput').addEventListener('input', filterMotions);
    
    // Filter selects
    document.getElementById('categoryFilter').addEventListener('change', filterMotions);
    document.getElementById('robotFilter').addEventListener('change', filterMotions);
    document.getElementById('sortBy').addEventListener('change', filterMotions);
    
    // Modal close button
    document.querySelector('.modal-close').addEventListener('click', closeModal);
    
    // Close modal when clicking outside
    document.getElementById('videoModal').addEventListener('click', (e) => {
        if (e.target.id === 'videoModal') {
            closeModal();
        }
    });
    
    // Close create modal when clicking outside
    document.getElementById('createModal').addEventListener('click', (e) => {
        if (e.target.id === 'createModal') {
            closeCreateModal();
        }
    });
    
    // ESC key to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal();
            closeCreateModal();
        }
    });
    
    // Create new motion button
    document.getElementById('createNewBtn').addEventListener('click', openCreateModal);
});

// Open create motion modal
function openCreateModal() {
    const modal = document.getElementById('createModal');
    modal.classList.add('active');
    
    // Clear form
    document.getElementById('newMotionName').value = '';
    document.getElementById('newMotionRobot').value = 'toddlerbot_2xc';
    
    // Focus on name input
    setTimeout(() => {
        document.getElementById('newMotionName').focus();
    }, 100);
}

// Close create motion modal
function closeCreateModal() {
    const modal = document.getElementById('createModal');
    modal.classList.remove('active');
}

// Create new motion
function createNewMotion() {
    const motionName = document.getElementById('newMotionName').value.trim();
    const robotName = document.getElementById('newMotionRobot').value;
    
    if (!motionName) {
        alert('Please enter a motion name');
        return;
    }
    
    // Validate motion name (lowercase, underscores only)
    if (!/^[a-z0-9_]+$/.test(motionName)) {
        alert('Motion name must contain only lowercase letters, numbers, and underscores');
        return;
    }
    
    // Close modal
    closeCreateModal();
    
    // Show loading message
    const loadingMsg = document.createElement('div');
    loadingMsg.style.cssText = 'position:fixed;top:20px;right:20px;background:#9c27b0;color:white;padding:15px 25px;border-radius:8px;z-index:10000;box-shadow:0 4px 6px rgba(0,0,0,0.3);font-family:system-ui;';
    loadingMsg.textContent = `Creating ${motionName}...`;
    document.body.appendChild(loadingMsg);
    
    // Call reload API with create flag
    const apiUrl = `http://localhost:8082/reload?motion=${encodeURIComponent(motionName)}&robot=${encodeURIComponent(robotName)}&create=true`;
    
    fetch(apiUrl)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                loadingMsg.style.background = '#4CAF50';
                const endTime = Date.now() + 10000;
                let countdownInterval = null;
                let launched = false;

                const finishLaunch = () => {
                    if (launched) {
                        return;
                    }
                    launched = true;
                    if (countdownInterval !== null) {
                        clearInterval(countdownInterval);
                    }
                    loadingMsg.innerHTML = `
                        <div style="font-weight:bold;">✅ Created ${motionName}</div>
                        <div style="font-size:12px;margin-top:4px;">Opening Viser now...</div>
                    `;

                    const opened = window.open('http://localhost:8080', '_blank');
                    if (!opened) {
                        loadingMsg.style.background = '#ff9800';
                        loadingMsg.innerHTML = `
                            <div style="font-weight:bold;">⚠️ Popup Blocked</div>
                            <div style="font-size:12px;margin-top:8px;">
                                <a href="http://localhost:8080" target="_blank" style="color:white;text-decoration:underline;">
                                    Click here to open Viser
                                </a>
                            </div>
                        `;
                    }

                    setTimeout(() => loadingMsg.remove(), 2000);
                };

                const updateCountdown = () => {
                    const remainingMs = endTime - Date.now();
                    const remainingSeconds = Math.max(0, Math.ceil(remainingMs / 1000));
                    loadingMsg.innerHTML = `
                        <div style="font-weight:bold;">✅ Created ${motionName}</div>
                        <div style="font-size:12px;margin-top:4px;">Opening in ${remainingSeconds}...</div>
                    `;
                    if (remainingSeconds <= 0) {
                        finishLaunch();
                    }
                };

                updateCountdown();
                countdownInterval = setInterval(updateCountdown, 1000);
            } else {
                loadingMsg.style.background = '#f44336';
                loadingMsg.textContent = `❌ Error: ${data.message}`;
                setTimeout(() => loadingMsg.remove(), 5000);
            }
        })
        .catch(error => {
            loadingMsg.style.background = '#f44336';
            loadingMsg.textContent = `❌ Connection Error: ${error.message}`;
            setTimeout(() => loadingMsg.remove(), 5000);
        });
}


