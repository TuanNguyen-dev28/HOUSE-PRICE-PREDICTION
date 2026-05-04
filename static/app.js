/* ══════════════════════════════════════════════════════════════
   House Price AI — Frontend Logic
   ══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initPredictionForm();
    loadDashboardStats();
    initScrollAnimations();
    initNavbarScroll();
    initLocationSelector();
    initLocationExplorer();
});

/* ─── Prediction Form ────────────────────────────────────────── */

function initPredictionForm() {
    const form = document.getElementById('predict-form');
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        await makePrediction();
    });
}

async function makePrediction() {
    const btnText = document.querySelector('.btn-text');
    const btnLoading = document.querySelector('.btn-loading');
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultContent = document.getElementById('result-content');
    const resultError = document.getElementById('result-error');

    // Hide previous results
    resultPlaceholder.style.display = 'none';
    resultContent.style.display = 'none';
    resultError.style.display = 'none';

    // Collect form data
    const areaVal = document.getElementById('area').value;
    const floorsVal = document.getElementById('floors').value;
    const bedroomsVal = document.getElementById('bedrooms').value;
    const bathroomsVal = document.getElementById('bathrooms').value;

    // Client-side validation for required fields
    const missingFields = [];
    if (!areaVal || areaVal === '') missingFields.push('Diện tích');
    if (!floorsVal || floorsVal === '') missingFields.push('Số tầng');
    if (!bedroomsVal || bedroomsVal === '') missingFields.push('Phòng ngủ');
    if (!bathroomsVal || bathroomsVal === '') missingFields.push('Phòng tắm');

    if (missingFields.length > 0) {
        resultError.style.display = 'block';
        document.getElementById('error-message').textContent =
            `Vui lòng nhập: ${missingFields.join(', ')}`;
        return;
    }

    // Validate numeric ranges
    const area = parseFloat(areaVal);
    if (isNaN(area) || area < 1 || area > 1000) {
        resultError.style.display = 'block';
        document.getElementById('error-message').textContent =
            'Diện tích phải từ 1 đến 1000 m²';
        return;
    }

    // Show loading state
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline-flex';

    const hiddenLocation = document.getElementById('location_select').value;
    const formData = {
        area: parseFloat(areaVal),
        frontage: parseFloat(document.getElementById('frontage').value) || 0,
        access_road: parseFloat(document.getElementById('access_road').value) || 0,
        floors: parseInt(floorsVal),
        bedrooms: parseInt(bedroomsVal),
        bathrooms: parseInt(bathroomsVal),
        house_direction: document.getElementById('house_direction').value,
        balcony_direction: document.getElementById('balcony_direction').value,
        legal_status: document.getElementById('legal_status').value,
        furniture_state: document.getElementById('furniture_state').value,
        address: hiddenLocation || document.getElementById('location_search_input').value || ''
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            // Show detailed validation errors from server
            let errorMsg = data.error || 'Dự đoán thất bại';
            if (data.details && Array.isArray(data.details)) {
                errorMsg = data.details.join('; ');
            }
            throw new Error(errorMsg);
        }

        // Show result with animation
        displayResult(data, formData);

    } catch (error) {
        console.error('Prediction error:', error);
        resultError.style.display = 'block';
        document.getElementById('error-message').textContent = error.message;
    } finally {
        btnText.style.display = 'inline';
        btnLoading.style.display = 'none';
    }
}

function displayResult(data, formData) {
    const resultContent = document.getElementById('result-content');
    resultContent.style.display = 'block';

    // Support both single model (price_billion_vnd) and ensemble (ensemble.price_billion_vnd)
    const priceValue = data.ensemble ? data.ensemble.price_billion_vnd : data.price_billion_vnd;
    const vndValue = data.ensemble ? data.ensemble.price_vnd : data.price_vnd;

    // Animate price counter
    const priceEl = document.getElementById('price-value');
    animateCounter(priceEl, 0, priceValue, 1200);

    // Show VND value
    const vndEl = document.getElementById('price-vnd');
    const vndFormatted = new Intl.NumberFormat('vi-VN').format(vndValue);
    vndEl.textContent = `≈ ${vndFormatted} VNĐ`;

    // Build summary
    const summaryEl = document.getElementById('result-summary');
    summaryEl.innerHTML = `
        <div class="summary-item">
            <span class="summary-label">Area</span>
            <span class="summary-value">${formData.area} m²</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Floors</span>
            <span class="summary-value">${formData.floors}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Bedrooms</span>
            <span class="summary-value">${formData.bedrooms}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Bathrooms</span>
            <span class="summary-value">${formData.bathrooms}</span>
        </div>
    `;

    // Show model mode if ensemble
    if (data.ensemble && data.mode) {
        const modeEl = document.getElementById('result-mode');
        if (modeEl) {
            modeEl.textContent = `Mode: ${data.mode}`;
            modeEl.style.display = 'block';
        }
    }

    // Scroll result into view
    document.getElementById('result-card').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

/* ─── Animated Counter ───────────────────────────────────────── */

function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    const diff = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = start + diff * eased;

        element.textContent = current.toFixed(2);

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}

/* ─── Dashboard ──────────────────────────────────────────────── */

async function loadDashboardStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        if (stats.error) {
            console.error('Stats error:', stats.error);
            return;
        }

        // Update stat cards
        document.getElementById('dash-total').textContent = 
            new Intl.NumberFormat().format(stats.total_properties);
        document.getElementById('dash-avg-price').textContent = stats.price.mean;
        document.getElementById('dash-avg-area').textContent = stats.area.mean;
        document.getElementById('dash-max-price').textContent = stats.price.max;

        // Render charts
        if (stats.feature_importances) {
            renderFeatureChart(stats.feature_importances);
        }
        if (stats.price_distribution) {
            renderPriceChart(stats.price_distribution);
        }
        if (stats.area_distribution) {
            renderAreaChart(stats.area_distribution);
        }

    } catch (error) {
        console.error('Failed to load dashboard stats:', error);
    }
}

/* ─── Charts ─────────────────────────────────────────────────── */

const chartColors = {
    indigo: 'rgba(99, 102, 241, 0.8)',
    violet: 'rgba(139, 92, 246, 0.8)',
    cyan: 'rgba(6, 182, 212, 0.8)',
    emerald: 'rgba(16, 185, 129, 0.8)',
    amber: 'rgba(245, 158, 11, 0.8)',
    indigoBg: 'rgba(99, 102, 241, 0.15)',
    violetBg: 'rgba(139, 92, 246, 0.15)',
    cyanBg: 'rgba(6, 182, 212, 0.15)',
};

const chartDefaults = {
    color: '#94a3b8',
    borderColor: 'rgba(255, 255, 255, 0.06)',
};

function renderFeatureChart(data) {
    const ctx = document.getElementById('chart-features').getContext('2d');

    // Clean up feature names for display
    const labels = data.features.slice(0, 10).map(f => {
        return f.replace('_Encoded', '').replace('_', ': ');
    });

    const values = data.importances.slice(0, 10);

    // Generate gradient colors
    const colors = values.map((_, i) => {
        const t = i / values.length;
        const r = Math.round(99 + (6 - 99) * t);
        const g = Math.round(102 + (182 - 102) * t);
        const b = Math.round(241 + (212 - 241) * t);
        return `rgba(${r}, ${g}, ${b}, 0.8)`;
    });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance Score',
                data: values,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1')),
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `Importance: ${(ctx.raw * 100).toFixed(1)}%`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: chartDefaults.borderColor },
                    ticks: { color: chartDefaults.color, font: { size: 11 } },
                },
                y: {
                    grid: { display: false },
                    ticks: { color: chartDefaults.color, font: { size: 12, weight: '500' } },
                }
            },
            animation: {
                duration: 1500,
                easing: 'easeOutQuart'
            }
        }
    });
}

function renderPriceChart(data) {
    const ctx = document.getElementById('chart-price').getContext('2d');

    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.4)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0.02)');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels.map(l => l + ' tỷ'),
            datasets: [{
                label: 'Number of Properties',
                data: data.counts,
                backgroundColor: gradient,
                borderColor: chartColors.indigo,
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `${ctx.raw.toLocaleString()} properties`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: chartDefaults.color, font: { size: 11 } },
                },
                y: {
                    grid: { color: chartDefaults.borderColor },
                    ticks: { color: chartDefaults.color },
                }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' }
        }
    });
}

function renderAreaChart(data) {
    const ctx = document.getElementById('chart-area').getContext('2d');

    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.4)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0.02)');

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.labels.map(l => l + ' m²'),
            datasets: [{
                label: 'Number of Properties',
                data: data.counts,
                backgroundColor: gradient,
                borderColor: chartColors.cyan,
                borderWidth: 1,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#f1f5f9',
                    bodyColor: '#94a3b8',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 8,
                    displayColors: false,
                    callbacks: {
                        label: (ctx) => `${ctx.raw.toLocaleString()} properties`
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { color: chartDefaults.color, font: { size: 11 } },
                },
                y: {
                    grid: { color: chartDefaults.borderColor },
                    ticks: { color: chartDefaults.color },
                }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' }
        }
    });
}

/* ─── Scroll Animations ──────────────────────────────────────── */

function initScrollAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    // Add animation classes to elements
    document.querySelectorAll('.glass-card, .stat-card, .about-card').forEach((el, i) => {
        el.classList.add('animate-in');
        el.style.transitionDelay = `${i * 0.08}s`;
        observer.observe(el);
    });
}

/* ─── Navbar Scroll Effect ───────────────────────────────────── */

function initNavbarScroll() {
    const navbar = document.getElementById('navbar');
    let lastScroll = 0;

    window.addEventListener('scroll', () => {
        const currentScroll = window.scrollY;

        if (currentScroll > 100) {
            navbar.style.background = 'rgba(10, 14, 26, 0.95)';
            navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
        } else {
            navbar.style.background = 'rgba(10, 14, 26, 0.8)';
            navbar.style.boxShadow = 'none';
        }

        lastScroll = currentScroll;
    });

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
}

/* ─── Unaccent Helper ───────────────────────────────────────── */

function removeAccents(str) {
    return str.normalize('NFD').replace(/[\u0300-\u036f]/g, '')
              .replace(/đ/g, 'd').replace(/Đ/g, 'D');
}

/* ─── Location Selector ─────────────────────────────────────── */

let cachedLocations = [];

async function initLocationSelector() {
    const searchInput = document.getElementById('location_search_input');
    const dropdown = document.getElementById('location_dropdown');
    const hiddenInput = document.getElementById('location_select');
    const hint = document.getElementById('location-hint');

    // Load locations once
    try {
        const response = await fetch('/api/locations');
        const data = await response.json();
        cachedLocations = data.locations || [];
    } catch (error) {
        console.error('Failed to load locations:', error);
    }

    // Search input handler
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.trim().toLowerCase();
        const queryNoAccent = removeAccents(query);
        
        if (query.length < 2) {
            dropdown.style.display = 'none';
            return;
        }

        // Filter locations (supports both accented and unaccented search)
        const filtered = cachedLocations.filter(loc => {
            const locLower = loc.location.toLowerCase();
            return locLower.includes(query) || removeAccents(locLower).includes(queryNoAccent);
        }).slice(0, 10);

        if (filtered.length === 0) {
            dropdown.innerHTML = '<div class="location-item">Không tìm thấy khu vực</div>';
        } else {
            dropdown.innerHTML = filtered.map(loc => `
                <div class="location-item" data-location="${loc.location}">
                    <span class="location-name">${loc.location}</span>
                    <span class="location-count">${loc.count} tin đăng</span>
                </div>
            `).join('');

            // Add click handlers
            dropdown.querySelectorAll('.location-item[data-location]').forEach(item => {
                item.addEventListener('click', () => {
                    const location = item.dataset.location;
                    searchInput.value = location;
                    hiddenInput.value = location;
                    dropdown.style.display = 'none';
                    hint.textContent = `Đã chọn: ${location}`;
                });
            });
        }

        dropdown.style.display = 'block';
    });

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.style.display = 'none';
        }
    });
}

/* ─── Location Explorer ─────────────────────────────────────── */

async function initLocationExplorer() {
    const searchInput = document.getElementById('location-explorer-search');
    const clearBtn = document.getElementById('search-clear-btn');
    const resultsContainer = document.getElementById('location-results');
    const detailOverlay = document.getElementById('location-detail-overlay');
    const closeDetailBtn = document.getElementById('detail-close-btn');

    // Load initial locations
    await loadLocations();

    // Search handler
    searchInput.addEventListener('input', async (e) => {
        const query = e.target.value.trim();
        
        if (query.length >= 2) {
            clearBtn.style.display = 'block';
            await loadLocations(query);
        } else {
            clearBtn.style.display = 'none';
            await loadLocations();
        }
    });

    // Clear search
    clearBtn.addEventListener('click', async () => {
        searchInput.value = '';
        clearBtn.style.display = 'none';
        await loadLocations();
    });

    // Close detail modal
    closeDetailBtn.addEventListener('click', () => {
        detailOverlay.style.display = 'none';
    });

    detailOverlay.addEventListener('click', (e) => {
        if (e.target === detailOverlay) {
            detailOverlay.style.display = 'none';
        }
    });

    async function loadLocations(query = '') {
        try {
            const url = query ? `/api/locations?q=${encodeURIComponent(query)}` : '/api/locations';
            const response = await fetch(url);
            const data = await response.json();
            
            resultsContainer.innerHTML = data.locations.slice(0, 20).map(loc => `
                <div class="location-card glass-card" data-location="${loc.location}">
                    <div class="location-card-header">
                        <h4>${loc.location}</h4>
                        <span class="location-badge">${loc.count} tin</span>
                    </div>
                    <div class="location-card-stats">
                        <div class="loc-stat">
                            <span class="loc-stat-value">${loc.avg_price.toFixed(1)} tỷ</span>
                            <span class="loc-stat-label">Giá TB</span>
                        </div>
                        <div class="loc-stat">
                            <span class="loc-stat-value">${loc.avg_area.toFixed(0)} m²</span>
                            <span class="loc-stat-label">DT TB</span>
                        </div>
                    </div>
                    <button class="location-detail-btn" data-location="${loc.location}">Xem chi tiết</button>
                </div>
            `).join('');

            // Add click handlers for detail buttons
            resultsContainer.querySelectorAll('.location-detail-btn').forEach(btn => {
                btn.addEventListener('click', async () => {
                    const location = btn.dataset.location;
                    await showLocationDetail(location);
                });
            });

            // Add click handlers for location cards
            resultsContainer.querySelectorAll('.location-card').forEach(card => {
                card.addEventListener('click', async (e) => {
                    if (e.target.classList.contains('location-detail-btn')) return;
                    const location = card.dataset.location;
                    await showLocationDetail(location);
                });
            });

        } catch (error) {
            console.error('Failed to load locations:', error);
            resultsContainer.innerHTML = '<p class="error">Không thể tải dữ liệu khu vực</p>';
        }
    }

    async function showLocationDetail(location) {
        try {
            const response = await fetch(`/api/location-detail/${encodeURIComponent(location)}`);
            const data = await response.json();

            document.getElementById('location-detail-content').innerHTML = `
                <div class="detail-header">
                    <h3>${data.location}</h3>
                    <p>${data.count} bất động sản trong khu vực</p>
                </div>
                <div class="detail-stats">
                    <div class="detail-stat">
                        <span class="detail-stat-value">${data.avg_price} tỷ</span>
                        <span class="detail-stat-label">Giá trung bình</span>
                    </div>
                    <div class="detail-stat">
                        <span class="detail-stat-value">${data.median_price} tỷ</span>
                        <span class="detail-stat-label">Giá trung vị</span>
                    </div>
                    <div class="detail-stat">
                        <span class="detail-stat-value">${data.min_price} - ${data.max_price} tỷ</span>
                        <span class="detail-stat-label">Khoảng giá</span>
                    </div>
                    <div class="detail-stat">
                        <span class="detail-stat-value">${data.avg_area} m²</span>
                        <span class="detail-stat-label">Diện tích TB</span>
                    </div>
                </div>
                <div class="detail-chart">
                    <h4>Phân bố giá (tỷ VNĐ)</h4>
                    <div class="detail-bars">
                        ${data.price_distribution.labels.map((label, i) => `
                            <div class="detail-bar-container">
                                <div class="detail-bar" style="width: ${Math.max(5, (data.price_distribution.counts[i] / Math.max(...data.price_distribution.counts)) * 100)}%"></div>
                                <span class="detail-bar-label">${label}</span>
                                <span class="detail-bar-count">${data.price_distribution.counts[i]}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                ${data.representative_address ? `
                <div class="detail-address">
                    <h4>Địa chỉ mẫu</h4>
                    <p>${data.representative_address}</p>
                </div>
                ` : ''}
            `;

            detailOverlay.style.display = 'flex';

        } catch (error) {
            console.error('Failed to load location detail:', error);
        }
    }
}
