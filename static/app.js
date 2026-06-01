/* ══════════════════════════════════════════════════════════════
   House Price AI — Frontend Logic
   ══════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initPredictionForm();
    loadDashboardStats();
    initLocationSelector();
    initLocationExplorer();
    initNavbarScroll();
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

    // Client-side validation
    const missingFields = [];
    if (!areaVal) missingFields.push('Diện tích');
    if (!floorsVal) missingFields.push('Số tầng');
    if (!bedroomsVal) missingFields.push('Phòng ngủ');
    if (!bathroomsVal) missingFields.push('Phòng tắm');

    if (missingFields.length > 0) {
        resultError.style.display = 'block';
        document.getElementById('error-message').textContent = `Vui lòng nhập: ${missingFields.join(', ')}`;
        return;
    }

    // Show loading state
    btnText.style.display = 'none';
    btnLoading.style.display = 'inline-flex';

    const formData = {
        area: parseFloat(areaVal),
        frontage: parseFloat(document.getElementById('frontage').value) || 0,
        access_road: parseFloat(document.getElementById('access_road').value) || 0,
        floors: parseInt(floorsVal),
        bedrooms: parseInt(bedroomsVal),
        bathrooms: parseInt(bathroomsVal),
        property_type: document.getElementById('property_type').value,
        house_direction: document.getElementById('house_direction').value,
        balcony_direction: document.getElementById('balcony_direction').value,
        legal_status: document.getElementById('legal_status').value,
        furniture_state: document.getElementById('furniture_state').value,
        address: document.getElementById('location_select').value || ''
    };

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || data.details?.join('; ') || 'Dự đoán thất bại');
        }

        displayResult(data, formData);

    } catch (error) {
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

    const priceValue = data.price_billion_vnd || data.ensemble?.price_billion_vnd || 0;
    const vndValue = data.price_vnd || data.ensemble?.price_vnd || 0;

    // Animate price counter
    const priceEl = document.getElementById('price-value');
    animateCounter(priceEl, 0, priceValue, 1200);

    // Show VND value
    const vndEl = document.getElementById('price-vnd');
    vndEl.textContent = `≈ ${new Intl.NumberFormat('vi-VN').format(vndValue)} VNĐ`;

    // Build summary
    const summaryEl = document.getElementById('result-summary');
    summaryEl.innerHTML = `
        <div class="summary-item">
            <span class="summary-label">Kiểu nhà</span>
            <span class="summary-value">${formData.property_type}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Diện tích</span>
            <span class="summary-value">${formData.area} m²</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Số tầng</span>
            <span class="summary-value">${formData.floors}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Phòng ngủ</span>
            <span class="summary-value">${formData.bedrooms}</span>
        </div>
        <div class="summary-item">
            <span class="summary-label">Phòng tắm</span>
            <span class="summary-value">${formData.bathrooms}</span>
        </div>
    `;

    document.getElementById('result-card').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    const diff = end - start;

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        element.textContent = (start + diff * eased).toFixed(2);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

/* ─── Dashboard ──────────────────────────────────────────────── */

async function loadDashboardStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();

        if (stats.error) return;

        document.getElementById('dash-total').textContent = 
            new Intl.NumberFormat().format(stats.total_properties);
        document.getElementById('dash-avg-price').textContent = stats.price.mean;
        document.getElementById('dash-avg-area').textContent = stats.area.mean;
        document.getElementById('dash-max-price').textContent = stats.price.max;

        if (stats.feature_importances) renderFeatureChart(stats.feature_importances);
        if (stats.price_distribution) renderPriceChart(stats.price_distribution);
        if (stats.area_distribution) renderAreaChart(stats.area_distribution);

    } catch (error) {
        console.error('Failed to load dashboard stats:', error);
    }
}

const chartColors = {
    indigo: 'rgba(99, 102, 241, 0.8)',
    cyan: 'rgba(6, 182, 212, 0.8)',
};

const chartDefaults = {
    color: '#94a3b8',
    borderColor: 'rgba(255, 255, 255, 0.06)',
};

function renderFeatureChart(data) {
    const ctx = document.getElementById('chart-features').getContext('2d');
    const labels = data.features.slice(0, 10).map(f => f.replace('_Encoded', '').replace('_', ': '));
    const values = data.importances.slice(0, 10);

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance Score',
                data: values,
                backgroundColor: 'rgba(99, 102, 241, 0.8)',
                borderColor: 'rgba(99, 102, 241, 1)',
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
                }
            },
            scales: {
                x: { grid: { color: chartDefaults.borderColor }, ticks: { color: chartDefaults.color } },
                y: { grid: { display: false }, ticks: { color: chartDefaults.color } }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' }
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
                label: 'Số BĐS',
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
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: chartDefaults.color } },
                y: { grid: { color: chartDefaults.borderColor }, ticks: { color: chartDefaults.color } }
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
                label: 'Số BĐS',
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
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: chartDefaults.color } },
                y: { grid: { color: chartDefaults.borderColor }, ticks: { color: chartDefaults.color } }
            },
            animation: { duration: 1500, easing: 'easeOutQuart' }
        }
    });
}

/* ─── Location Selector (3 autocomplete fields) ──────────────── */

/* ─── Location Selector (Hierarchical Dropdowns) ──────────────── */

let locationHierarchy = {};

async function initLocationSelector() {
    const districtSelect = document.getElementById('district_select');
    const wardSelect = document.getElementById('ward_select');
    const streetSelect = document.getElementById('street_select');
    const hint = document.getElementById('location-hint');
    const hiddenLocation = document.getElementById('location_select');

    // 1. Load location hierarchy data from JSON file
    try {
        const response = await fetch('/location_hierarchy.json');
        locationHierarchy = await response.json();
        
        // 2. Populate District dropdown
        populateDropdown(districtSelect, Object.keys(locationHierarchy), '-- Chọn Quận/Huyện --');
    } catch (error) {
        console.error('Failed to load location hierarchy:', error);
        hint.textContent = '❌ Lỗi tải danh mục địa điểm, vui lòng tải lại trang.';
        return;
    }

    // 3. District change listener
    districtSelect.addEventListener('change', () => {
        const selectedDistrict = districtSelect.value;
        
        // Reset Ward and Street
        resetDropdown(wardSelect, '-- Chọn Phường/Xã --');
        resetDropdown(streetSelect, '-- Chọn Tên Đường --');
        hiddenLocation.value = '';
        
        if (selectedDistrict) {
            // Enable Ward select and populate it
            wardSelect.disabled = false;
            const wards = Object.keys(locationHierarchy[selectedDistrict] || {});
            populateDropdown(wardSelect, wards, '-- Chọn Phường/Xã --');
            
            hint.textContent = `Đã chọn: ${selectedDistrict}`;
            hiddenLocation.value = `${selectedDistrict}, TP.HCM`;
        } else {
            wardSelect.disabled = true;
            streetSelect.disabled = true;
            hint.textContent = 'Chọn vị trí theo thứ tự để bắt đầu định giá';
        }
    });

    // 4. Ward change listener
    wardSelect.addEventListener('change', () => {
        const selectedDistrict = districtSelect.value;
        const selectedWard = wardSelect.value;
        
        // Reset Street
        resetDropdown(streetSelect, '-- Chọn Tên Đường --');
        
        if (selectedWard) {
            // Enable Street select and populate it
            streetSelect.disabled = false;
            const streets = locationHierarchy[selectedDistrict][selectedWard] || [];
            populateDropdown(streetSelect, streets, '-- Chọn Tên Đường --');
            
            hint.textContent = `Đã chọn: ${selectedDistrict} ➔ ${selectedWard}`;
            hiddenLocation.value = `${selectedWard}, ${selectedDistrict}, TP.HCM`;
        } else {
            streetSelect.disabled = true;
            hint.textContent = `Đã chọn: ${selectedDistrict}`;
            hiddenLocation.value = `${selectedDistrict}, TP.HCM`;
        }
    });

    // 5. Street change listener
    streetSelect.addEventListener('change', () => {
        const selectedDistrict = districtSelect.value;
        const selectedWard = wardSelect.value;
        const selectedStreet = streetSelect.value;
        
        if (selectedStreet) {
            hint.textContent = `Đã chọn: ${selectedDistrict} ➔ ${selectedWard} ➔ ${selectedStreet}`;
            hiddenLocation.value = `${selectedStreet}, ${selectedWard}, ${selectedDistrict}, TP.HCM`;
        } else {
            hint.textContent = `Đã chọn: ${selectedDistrict} ➔ ${selectedWard}`;
            hiddenLocation.value = `${selectedWard}, ${selectedDistrict}, TP.HCM`;
        }
    });
}

function populateDropdown(selectElement, itemsList, defaultText) {
    selectElement.innerHTML = `<option value="">${defaultText}</option>`;
    itemsList.forEach(item => {
        const option = document.createElement('option');
        option.value = item;
        option.textContent = item;
        selectElement.appendChild(option);
    });
}

function resetDropdown(selectElement, defaultText) {
    selectElement.innerHTML = `<option value="">${defaultText}</option>`;
    selectElement.disabled = true;
}

function removeAccents(str) {
    return str.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/g, 'd').replace(/Đ/g, 'D');
}

/* ─── Location Explorer ──────────────────────────────────────── */

let allLocations = [];
let currentSort = 'count';
let currentPage = 1;
const ITEMS_PER_PAGE = 9;

async function initLocationExplorer() {
    const searchInput = document.getElementById('location-search');
    const clearBtn = document.getElementById('clear-search');
    const modal = document.getElementById('district-modal');
    const modalClose = document.getElementById('modal-close');
    const filterTabs = document.querySelectorAll('.filter-tab');

    // Load initial data
    await loadLocations();

    // Search handler
    searchInput.addEventListener('input', async (e) => {
        const query = e.target.value.trim();
        clearBtn.style.display = query.length > 0 ? 'block' : 'none';
        currentPage = 1;
        await loadLocations(query);
    });

    // Clear search
    clearBtn.addEventListener('click', async () => {
        searchInput.value = '';
        clearBtn.style.display = 'none';
        currentPage = 1;
        await loadLocations();
    });

    // Filter tabs
    filterTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            filterTabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            currentSort = tab.dataset.sort;
            currentPage = 1;
            renderDistrictGrid(allLocations);
        });
    });

    // Pagination handlers
    document.getElementById('prev-page').addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            renderDistrictGrid(allLocations);
        }
    });

    document.getElementById('next-page').addEventListener('click', () => {
        const totalPages = Math.ceil(allLocations.length / ITEMS_PER_PAGE);
        if (currentPage < totalPages) {
            currentPage++;
            renderDistrictGrid(allLocations);
        }
    });

    // Close modal
    modalClose.addEventListener('click', () => modal.classList.remove('active'));
    modal.querySelector('.modal-backdrop').addEventListener('click', () => modal.classList.remove('active'));
}

async function loadLocations(query = '') {
    try {
        const url = query ? `/api/locations?q=${encodeURIComponent(query)}` : '/api/locations';
        const response = await fetch(url);
        const data = await response.json();

        allLocations = data.locations || [];
        currentPage = 1;

        // Update summary
        document.getElementById('total-districts').textContent = allLocations.length;

        if (allLocations.length > 0) {
            const avgPrice = (allLocations.reduce((sum, loc) => sum + loc.avg_price, 0) / allLocations.length).toFixed(1);
            document.getElementById('avg-price-district').textContent = avgPrice;

            const highest = allLocations.reduce((max, loc) => loc.avg_price > max.avg_price ? loc : max);
            document.getElementById('highest-district').textContent = highest.location.split(',')[0];
        }
        
        renderDistrictGrid(allLocations);
    } catch (error) {
        console.error('Failed to load locations:', error);
    }
}

function renderDistrictGrid(locations) {
    const grid = document.getElementById('district-grid');
    const prevBtn = document.getElementById('prev-page');
    const nextBtn = document.getElementById('next-page');
    const paginationNumbers = document.getElementById('pagination-numbers');

    // Sort locations
    let sorted = [...locations];
    if (currentSort === 'price') {
        sorted.sort((a, b) => b.avg_price - a.avg_price);
    } else if (currentSort === 'price_low') {
        sorted.sort((a, b) => a.avg_price - b.avg_price);
    } else {
        sorted.sort((a, b) => b.count - a.count);
    }

    if (sorted.length === 0) {
        grid.innerHTML = '<p style="text-align:center; color: var(--text-muted); grid-column: 1/-1; padding: 60px;">Không tìm thấy khu vực nào</p>';
        paginationNumbers.innerHTML = '';
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        return;
    }

    // Calculate pagination
    const totalPages = Math.ceil(sorted.length / ITEMS_PER_PAGE);
    currentPage = Math.min(currentPage, totalPages);
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
    const endIndex = startIndex + ITEMS_PER_PAGE;
    const pageItems = sorted.slice(startIndex, endIndex);

    // Render grid items
    grid.innerHTML = pageItems.map(loc => `
        <div class="district-card" data-location="${loc.location}">
            <div class="district-card-header">
                <span class="district-name">${loc.location.split(',')[0]}</span>
                <span class="district-count">${loc.count} tin</span>
            </div>
            <div class="district-price">
                <span class="district-price-value">${loc.avg_price.toFixed(1)}</span>
                <span class="district-price-unit">tỷ VNĐ</span>
            </div>
            <div class="district-stats">
                <div class="district-stat">
                    <span class="district-stat-value">${loc.median_price.toFixed(1)}</span>
                    <span class="district-stat-label">Giá trung vị</span>
                </div>
                <div class="district-stat">
                    <span class="district-stat-value">${loc.avg_area.toFixed(0)}</span>
                    <span class="district-stat-label">DT TB (m²)</span>
                </div>
            </div>
            <div class="district-card-footer">
                <span class="district-range">${loc.min_price.toFixed(1)} - ${loc.max_price.toFixed(1)} tỷ</span>
                <span class="district-view-btn">Chi tiết →</span>
            </div>
        </div>
    `).join('');

    // Add click handlers
    grid.querySelectorAll('.district-card').forEach(card => {
        card.addEventListener('click', () => showDistrictDetail(card.dataset.location));
    });

    // Update pagination buttons
    prevBtn.disabled = currentPage === 1;
    nextBtn.disabled = currentPage === totalPages;

    // Render pagination numbers
    renderPaginationNumbers(totalPages);
}

function renderPaginationNumbers(totalPages) {
    const container = document.getElementById('pagination-numbers');
    container.innerHTML = '';

    if (totalPages <= 1) return;

    const maxVisible = 5;
    let startPage = Math.max(1, currentPage - Math.floor(maxVisible / 2));
    let endPage = Math.min(totalPages, startPage + maxVisible - 1);

    if (endPage - startPage < maxVisible - 1) {
        startPage = Math.max(1, endPage - maxVisible + 1);
    }

    // First page
    if (startPage > 1) {
        container.innerHTML += `<button class="page-num" data-page="1">1</button>`;
        if (startPage > 2) {
            container.innerHTML += `<span class="page-ellipsis">...</span>`;
        }
    }

    // Middle pages
    for (let i = startPage; i <= endPage; i++) {
        const activeClass = i === currentPage ? ' active' : '';
        container.innerHTML += `<button class="page-num${activeClass}" data-page="${i}">${i}</button>`;
    }

    // Last page
    if (endPage < totalPages) {
        if (endPage < totalPages - 1) {
            container.innerHTML += `<span class="page-ellipsis">...</span>`;
        }
        container.innerHTML += `<button class="page-num" data-page="${totalPages}">${totalPages}</button>`;
    }

    // Add click handlers
    container.querySelectorAll('.page-num').forEach(btn => {
        btn.addEventListener('click', () => {
            currentPage = parseInt(btn.dataset.page);
            renderDistrictGrid(allLocations);
        });
    });
}

async function showDistrictDetail(location) {
    const modal = document.getElementById('district-modal');
    const modalBody = document.getElementById('modal-body');
    
    modalBody.innerHTML = '<div style="text-align:center; padding:60px; color:var(--text-muted);">Đang tải...</div>';
    modal.classList.add('active');

    try {
        const response = await fetch(`/api/location-detail/${encodeURIComponent(location)}`);
        const data = await response.json();

        const maxCount = Math.max(...data.price_distribution.counts);

        modalBody.innerHTML = `
            <div class="modal-header">
                <h2 class="modal-title">${data.location.split(',')[0]}</h2>
                <span class="modal-badge">${data.count} bất động sản</span>
            </div>
            
            <div class="modal-price-hero">
                <div class="modal-price-value">${data.avg_price.toFixed(2)}</div>
                <div class="modal-price-unit">tỷ VNĐ</div>
                <span class="modal-price-label">Giá trung bình</span>
            </div>
            
            <div class="modal-stats-grid">
                <div class="modal-stat">
                    <span class="modal-stat-icon">📊</span>
                    <span class="modal-stat-value">${data.median_price.toFixed(2)}</span>
                    <span class="modal-stat-label">Giá trung vị</span>
                </div>
                <div class="modal-stat">
                    <span class="modal-stat-icon">📉</span>
                    <span class="modal-stat-value">${data.min_price.toFixed(1)}</span>
                    <span class="modal-stat-label">Giá thấp nhất</span>
                </div>
                <div class="modal-stat">
                    <span class="modal-stat-icon">📈</span>
                    <span class="modal-stat-value">${data.max_price.toFixed(1)}</span>
                    <span class="modal-stat-label">Giá cao nhất</span>
                </div>
                <div class="modal-stat">
                    <span class="modal-stat-icon">📐</span>
                    <span class="modal-stat-value">${data.avg_area.toFixed(0)}</span>
                    <span class="modal-stat-label">DT TB (m²)</span>
                </div>
            </div>
            
            <div class="modal-chart-section">
                <h4>Phân bố giá (tỷ VNĐ)</h4>
                <div class="modal-bars">
                    ${data.price_distribution.labels.map((label, i) => `
                        <div class="modal-bar-row">
                            <span class="modal-bar-label">${label}</span>
                            <div class="modal-bar" style="width: ${(data.price_distribution.counts[i] / maxCount) * 100}%"></div>
                            <span style="font-size:0.8rem; color:var(--text-muted);">${data.price_distribution.counts[i]}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            ${data.representative_address ? `
            <div class="modal-address">
                <h4>Địa chỉ mẫu</h4>
                <p>${data.representative_address}</p>
            </div>
            ` : ''}
        `;

    } catch (error) {
        modalBody.innerHTML = '<p style="text-align:center; color:#f87171; padding:40px;">Lỗi khi tải dữ liệu</p>';
    }
}

/* ─── Navbar Scroll Effect ───────────────────────────────────── */

function initNavbarScroll() {
    const navbar = document.getElementById('navbar');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 100) {
            navbar.style.background = 'rgba(10, 14, 26, 0.95)';
            navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
        } else {
            navbar.style.background = 'rgba(10, 14, 26, 0.8)';
            navbar.style.boxShadow = 'none';
        }
    });

    // Smooth scroll for nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.querySelector(link.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth' });
        });
    });
}
