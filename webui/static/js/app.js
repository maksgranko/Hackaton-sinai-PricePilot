const CONFIG = window.__WEBUI_CONFIG__ || {};
const RAW_BACKEND_BASE = (CONFIG.backendBase || "").trim();
const BACKEND_BASE =
  RAW_BACKEND_BASE && RAW_BACKEND_BASE !== "/"
    ? RAW_BACKEND_BASE.replace(/\/+$/, "")
    : "";
const TOKEN_PATH = CONFIG.tokenPath || "/auth/token";
const PRICING_PATH = CONFIG.pricingPath || "/api/v1/orders/price-recommendation";
const TOKEN_ENDPOINT = `${BACKEND_BASE}${TOKEN_PATH.startsWith("/") ? TOKEN_PATH : `/${TOKEN_PATH}`}`;
const API_ENDPOINT = `${BACKEND_BASE}${PRICING_PATH.startsWith("/") ? PRICING_PATH : `/${PRICING_PATH}`}`;
const INCLUDE_CREDENTIALS = CONFIG.includeCredentials === true;

const DEMO_ACCOUNT = {
  username: CONFIG.username || "demo@example.com",
  password: CONFIG.password || "demo",
};

const BASE_ORDER = {
  order_timestamp: Math.floor(Date.now() / 1000),
  distance_in_meters: 12000,
  duration_in_seconds: 1600,
  pickup_in_meters: 2000,
  pickup_in_seconds: 120,
  driver_rating: 4.8,
  platform: "android",
  price_start_local: 180,
  carname: "LADA",
  carmodel: "GRANTA",
  driver_reg_date: "2020-01-15",
  ...(CONFIG.orderDefaults || {}),
};

if (BASE_ORDER.price_start_local != null) {
  BASE_ORDER.price_start_local = Number(BASE_ORDER.price_start_local);
}
if (BASE_ORDER.driver_rating != null) {
  BASE_ORDER.driver_rating = parseFloat(BASE_ORDER.driver_rating);
}
if (BASE_ORDER.carname == null) {
  BASE_ORDER.carname = "";
}
if (BASE_ORDER.carmodel == null) {
  BASE_ORDER.carmodel = "";
}
if (BASE_ORDER.driver_reg_date == null) {
  BASE_ORDER.driver_reg_date = "";
}

const state = {
  token: null,
  data: null,
  order: { ...BASE_ORDER },
  priceMin: 0,
  priceMax: 0,
  priceStep: 5,
  ready: false,
  debugPanelVisible: false,
};

let pricePointer;
let priceScale;
let priceInput;
let isDragging = false;

function logAction(action) {
  console.log(`[ACTION] ${new Date().toLocaleTimeString()}: ${action}`);
}

function showLoading() {
  const loadingOverlay = document.getElementById("loading-overlay");
  if (loadingOverlay) {
    loadingOverlay.classList.add("visible");
  }
}

function hideLoading() {
  const loadingOverlay = document.getElementById("loading-overlay");
  if (loadingOverlay) {
    loadingOverlay.classList.remove("visible");
  }
}

async function requestToken(credentials = DEMO_ACCOUNT) {
  const payload = new URLSearchParams();
  payload.set("username", credentials.username);
  payload.set("password", credentials.password);

  const response = await fetch(TOKEN_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    credentials: INCLUDE_CREDENTIALS ? "include" : "same-origin",
    body: payload,
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(`Auth failed: ${text || response.statusText}`);
  }

  const tokenPayload = await response.json();
  return tokenPayload.access_token;
}

async function requestPricing(orderPayload) {
  if (!state.token) {
    state.token = await requestToken();
  }

  const response = await fetch(API_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${state.token}`,
    },
    credentials: INCLUDE_CREDENTIALS ? "include" : "same-origin",
    body: JSON.stringify(orderPayload),
  });

  if (!response.ok) {
    const text = await response.text();
    if (response.status === 401) {
      state.token = null;
    }
    throw new Error(text || `Pricing request failed: ${response.statusText}`);
  }

  return response.json();
}

function ensureDataReady() {
  if (!state.data) {
    throw new Error("PricePilot data не загружены.");
  }
}

function formatCurrency(value) {
  return `${Number(value).toFixed(0)}₽`;
}

function normalizeTimestamp(value) {
  if (value == null) {
    return Math.floor(Date.now() / 1000);
  }
  
  // If it's already a number (Unix timestamp), return it
  if (typeof value === 'number') {
    return Math.floor(value);
  }
  
  // If it's a string, try to parse it as a date
  if (typeof value === 'string') {
    const date = new Date(value);
    if (!isNaN(date.getTime())) {
      return Math.floor(date.getTime() / 1000);
    }
  }
  
  // Fallback to current timestamp
  return Math.floor(Date.now() / 1000);
}

function setDebugStatus(elementId, message, variant = "") {
  const el = document.getElementById(elementId);
  if (!el) return;
  el.textContent = message;
  el.classList.remove("success", "error");
  if (variant) {
    el.classList.add(variant);
  }
}

function computePriceBoundsFromData(data) {
  // Если есть зоны - используем их границы
  if (data?.zones && data.zones.length > 0) {
    const firstZone = data.zones[0];
    const lastZone = data.zones[data.zones.length - 1];
    
    const min = Number(firstZone.price_range.min);
    const zonesMax = Number(lastZone.price_range.max);
    
    // Расширяем диапазон на 50% выше последней зоны, 
    // чтобы показать красные/жёлтые зоны с низкой вероятностью
    const extension = (zonesMax - min) * 0.5;
    const max = zonesMax + Math.max(extension, 100); // минимум +100₽
    
    const stepCandidate = Number(data?.analysis?.price_increment ?? state.priceStep ?? 5);
    const step = stepCandidate > 0 ? stepCandidate : 5;
    
    return { min, max, step };
  }
  
  // Fallback на старую логику, если зон нет
  const rawMin =
    data?.analysis?.scan_range?.min ?? data?.analysis?.start_price ?? state.priceMin ?? 0;
  const rawMax = data?.analysis?.scan_range?.max;
  const fallbackMax = (() => {
    const base = data?.analysis?.start_price ?? rawMin;
    return base + 200;
  })();
  const min = Number.isFinite(Number(rawMin)) ? Number(rawMin) : 0;
  const maxCandidate = Number.isFinite(Number(rawMax)) ? Number(rawMax) : fallbackMax;
  const max = Math.max(min + 1, maxCandidate);
  const stepCandidate = Number(data?.analysis?.price_increment ?? state.priceStep ?? 5);
  const step = stepCandidate > 0 ? stepCandidate : 5;

  return { min, max, step };
}

function enrichData(data) {
  if (!data.price_probabilities) {
    data.price_probabilities = {};
  }
  if (!data.recommendations) {
    data.recommendations = [];
  }
  if (!data.zones) {
    data.zones = [];
  }
  return data;
}

function syncDebugControls() {
  const panel = document.getElementById("debug-panel");
  if (!panel) return;
}

function toggleDebugPanel(force) {
  const panel = document.getElementById("debug-panel");
  const overlay = document.getElementById("debug-overlay");
  if (!panel) return;
  const shouldShow =
    typeof force === "boolean" ? force : !state.debugPanelVisible;
  panel.classList.toggle("visible", shouldShow);
  if (overlay) {
    overlay.classList.toggle("visible", shouldShow);
  }
  state.debugPanelVisible = shouldShow;
  if (shouldShow) {
    syncDebugControls();
    logAction("Debug panel opened.");
  } else {
    logAction("Debug panel closed.");
  }
}

function clearJsonEditor() {
  const input = document.getElementById("debug-json-input");
  if (input) {
    input.value = "";
  }
  setDebugStatus("debug-json-status", "JSON cleared.", "");
}

async function applyJsonOverride() {
  const input = document.getElementById("debug-json-input");
  if (!input) return;
  const raw = input.value.trim();
  if (!raw) {
    setDebugStatus("debug-json-status", "Введите JSON.", "error");
    return;
  }
  try {
    const overrides = JSON.parse(raw);
    state.order = {
      ...state.order,
      ...overrides,
    };
    
    // Normalize order_timestamp (string datetime or number)
    state.order.order_timestamp = normalizeTimestamp(state.order.order_timestamp);
    
    if (state.order.price_start_local != null) {
      state.order.price_start_local = Number(state.order.price_start_local);
    }
    if (state.order.driver_rating != null) {
      state.order.driver_rating = parseFloat(state.order.driver_rating);
    }
    if (state.order.carname == null) {
      state.order.carname = "";
    }
    if (state.order.carmodel == null) {
      state.order.carmodel = "";
    }
    if (state.order.driver_reg_date == null) {
      state.order.driver_reg_date = "";
    }
    console.log("📊 JSON override applied:", state.order);
    setDebugStatus("debug-json-status", "JSON overrides applied. Fetching new data...", "success");
    logAction("JSON override applied via debugger.");
    const targetPrice =
      Number(state.order.price_start_local) || state.priceMin || BASE_ORDER.price_start_local;
    await activeBidUpdate(targetPrice);
    updateClientDetailsFromOrder();
    setDebugStatus("debug-json-status", "JSON overrides applied successfully!", "success");
  } catch (error) {
    console.error(error);
    setDebugStatus("debug-json-status", `Override failed: ${error.message}`, "error");
  }
}

async function refreshToken() {
  try {
    state.token = await requestToken();
    setDebugStatus("debug-token-status", "JWT refreshed successfully.", "success");
    logAction("JWT token refreshed via debugger.");
  } catch (error) {
    console.error(error);
    setDebugStatus("debug-token-status", `Token refresh failed: ${error.message}`, "error");
  }
}

function updateClientDetailsFromOrder() {
  const order = state.order;
  const detailsSpan = document.querySelector(".client-details span");
  if (!detailsSpan) return;
  
  const rating = order.driver_rating != null ? parseFloat(order.driver_rating).toFixed(1) : "5.0";
  const distanceKm = order.distance_in_meters != null 
    ? (order.distance_in_meters / 1000).toFixed(1) 
    : "12.0";
  const timeMin = order.duration_in_seconds != null 
    ? Math.round(order.duration_in_seconds / 60) 
    : "25";
  
  detailsSpan.innerHTML = `<i class="fas fa-star" style="color: var(--warning-color); margin-right: 3px"></i>${rating} Рейтинг | ${distanceKm} км | ${timeMin} мин`;
  
  // Обновляем время прибытия до клиента
  updatePickupTime();
}

function updatePickupTime() {
  const order = state.order;
  const pickupTimeText = document.getElementById("pickup-time-text");
  if (!pickupTimeText) return;
  
  if (order.pickup_in_seconds != null && order.pickup_in_seconds > 0) {
    const pickupMin = Math.round(order.pickup_in_seconds / 60);
    pickupTimeText.textContent = `~${pickupMin} мин до клиента`;
  } else if (order.pickup_in_meters != null && order.pickup_in_meters > 0) {
    // Fallback: примерно 30 км/ч в городе
    const estimatedSeconds = (order.pickup_in_meters / 1000) * (60 / 30) * 60;
    const pickupMin = Math.round(estimatedSeconds / 60);
    pickupTimeText.textContent = `~${pickupMin} мин до клиента`;
  } else {
    pickupTimeText.textContent = "~2 мин до клиента";
  }
  
  // Обновляем время в пути
  updateTripDuration();
}

function updateTripDuration() {
  const order = state.order;
  const tripDurationText = document.getElementById("trip-duration-text");
  if (!tripDurationText) return;
  
  let durationText = "";
  
  // Время в пути
  if (order.duration_in_seconds != null && order.duration_in_seconds > 0) {
    const durationMin = Math.round(order.duration_in_seconds / 60);
    durationText = `~${durationMin} мин в пути`;
  } else if (order.distance_in_meters != null && order.distance_in_meters > 0) {
    // Fallback: примерно 30 км/ч средняя скорость в городе
    const estimatedSeconds = (order.distance_in_meters / 1000) * (60 / 30) * 60;
    const durationMin = Math.round(estimatedSeconds / 60);
    durationText = `~${durationMin} мин в пути`;
  } else {
    durationText = "~25 мин в пути";
  }
  
  // Добавляем расстояние
  if (order.distance_in_meters != null && order.distance_in_meters > 0) {
    const distanceKm = (order.distance_in_meters / 1000).toFixed(1);
    durationText += ` • ${distanceKm} км`;
  }
  
  tripDurationText.textContent = durationText;
}

function hydrateSummaryPanels(data) {
  const optimal = data.optimal_price;
  
  // Сохраняем оптимальную цену в state для использования в setOptimalPrice
  state.optimalPrice = Math.round(optimal.price / state.priceStep) * state.priceStep;
  
  // Обновляем метки min/max цены - показываем расширенные границы
  document.getElementById("label-min-price").textContent = `${formatCurrency(
    state.priceMin
  )} (Мин)`;
  document.getElementById("label-avg-price").textContent = `${formatCurrency(
    (state.priceMin + state.priceMax) / 2
  )}`;
  document.getElementById("label-max-price").textContent = `${formatCurrency(
    state.priceMax
  )} (Макс)`;

  document.getElementById(
    "optimal-price-text"
  ).innerHTML = `<i class="fas fa-magic"></i>${state.optimalPrice}₽`;
  
  renderZoneMarkers(data);
}

function extractZoneColor(zoneName) {
  // Извлекаем цвет из названия зоны типа "zone_3_green" или "zone_1_red_low"
  if (zoneName.includes("green")) return "green";
  if (zoneName.includes("yellow")) return "yellow";
  if (zoneName.includes("red")) return "red";
  return "green"; // fallback
}

function translateZoneName(zoneName) {
  // Переводим названия зон на русский
  if (zoneName.includes("green")) return "Зелёная зона";
  if (zoneName.includes("yellow_low")) return "Жёлтая зона (низкая)";
  if (zoneName.includes("yellow_high")) return "Жёлтая зона (высокая)";
  if (zoneName.includes("red")) return "Красная зона";
  return zoneName; // fallback
}

function updatePriceScaleGradient(data) {
  if (!data || !data.zones) return;
  
  const scaleEl = document.getElementById("price-scale");
  if (!scaleEl) return;
  
  // Сортируем зоны по минимальной цене
  const sortedZones = data.zones.slice().sort((a, b) => a.price_range.min - b.price_range.min);
  
  // Создаем градиент на основе реальных позиций зон
  let gradientStops = [];
  
  sortedZones.forEach((zone, index) => {
    const minPos = ((zone.price_range.min - state.priceMin) / (state.priceMax - state.priceMin)) * 100;
    const maxPos = ((zone.price_range.max - state.priceMin) / (state.priceMax - state.priceMin)) * 100;
    
    // Определяем цвет зоны
    let color;
    if (zone.zone_name.includes("green")) {
      color = "#28a745"; // Зелёный
    } else if (zone.zone_name.includes("yellow")) {
      color = "#f39c12"; // Жёлтый
    } else if (zone.zone_name.includes("red")) {
      color = "#e74c3c"; // Красный
    } else {
      color = "#6c757d"; // Серый по умолчанию
    }
    
    // Добавляем точки градиента для начала и конца зоны
    gradientStops.push(`${color} ${minPos}%`);
    gradientStops.push(`${color} ${maxPos}%`);
  });
  
  // Если есть пустые области, заполняем их красным (низкая вероятность)
  if (sortedZones.length > 0) {
    const firstZone = sortedZones[0];
    const lastZone = sortedZones[sortedZones.length - 1];
    
    const firstMinPos = ((firstZone.price_range.min - state.priceMin) / (state.priceMax - state.priceMin)) * 100;
    const lastMaxPos = ((lastZone.price_range.max - state.priceMin) / (state.priceMax - state.priceMin)) * 100;
    
    // Добавляем красные области в начале и конце, если нужно
    if (firstMinPos > 0) {
      gradientStops.unshift("#c0392b 0%", "#c0392b " + firstMinPos + "%");
    }
    if (lastMaxPos < 100) {
      gradientStops.push("#c0392b " + lastMaxPos + "%", "#c0392b 100%");
    }
  } else {
    // Если зон нет, делаем всё красным
    gradientStops = ["#c0392b 0%", "#c0392b 100%"];
  }
  
  const gradient = `linear-gradient(to right, ${gradientStops.join(", ")})`;
  scaleEl.style.background = gradient;
}

function renderZoneMarkers(data) {
  if (!data || !data.zones) return;
  
  const scaleEl = document.getElementById("price-scale");
  if (!scaleEl) return;
  
  // Обновляем градиент в зависимости от зон
  updatePriceScaleGradient(data);
  
  // Удаляем старые маркеры зон и границы
  const oldMarkers = scaleEl.querySelectorAll(".zone-marker, .zone-boundary");
  oldMarkers.forEach(m => m.remove());
  
  // Создаём маркеры для каждой зоны
  data.zones.forEach((zone) => {
    const minPos = ((zone.price_range.min - state.priceMin) / (state.priceMax - state.priceMin)) * 100;
    const maxPos = ((zone.price_range.max - state.priceMin) / (state.priceMax - state.priceMin)) * 100;
    
    if (minPos < 0 || maxPos > 100 || minPos >= maxPos) return;
    
    const zoneColor = extractZoneColor(zone.zone_name);
    const marker = document.createElement("div");
    marker.className = `zone-marker zone-${zoneColor}`;
    marker.style.left = `${minPos}%`;
    marker.style.width = `${maxPos - minPos}%`;
    marker.title = `${translateZoneName(zone.zone_name)}: ${zone.price_range.min.toFixed(0)}-${zone.price_range.max.toFixed(0)}₽ (${zone.metrics.avg_probability_percent.toFixed(1)}%)`;
    
    scaleEl.appendChild(marker);
    
    // Добавляем визуальную границу в конце зоны
    const boundary = document.createElement("div");
    boundary.className = "zone-boundary";
    boundary.style.left = `${maxPos}%`;
    boundary.title = `Граница зоны: ${zone.price_range.max.toFixed(0)}₽`;
    scaleEl.appendChild(boundary);
  });
  
  // Если зон меньше 5, всё что после последней зоны - красная зона
  if (data.zones.length > 0) {
    const lastZone = data.zones[data.zones.length - 1];
    const lastZoneEnd = ((lastZone.price_range.max - state.priceMin) / (state.priceMax - state.priceMin)) * 100;
    
    if (lastZoneEnd < 100) {
      const redMarker = document.createElement("div");
      redMarker.className = "zone-marker zone-red";
      redMarker.style.left = `${lastZoneEnd}%`;
      redMarker.style.width = `${100 - lastZoneEnd}%`;
      redMarker.title = `Красная зона: ${lastZone.price_range.max.toFixed(0)}-${state.priceMax.toFixed(0)}₽ (низкая вероятность)`;
      scaleEl.appendChild(redMarker);
    }
  }
}

function updatePointerStyle(zone) {
  const pointer = pricePointer;
  const colorMap = {
    green: "var(--drivee-green)",
    yellow: "var(--warning-color)",
    red: "var(--danger-color)",
  };
  const color = colorMap[zone] || "var(--drivee-green)";

  pointer.style.borderColor = color;
  pointer.style.boxShadow = `0 0 10px ${color}`;
}

function interpolateProbability(price, zone, prevZone, nextZone) {
  const zoneMin = zone.price_range.min;
  const zoneMax = zone.price_range.max;
  const zoneProb = Number(zone.metrics.avg_normalized_probability_percent);
  
  // Позиция внутри зоны (0 = начало, 1 = конец)
  const positionInZone = (price - zoneMin) / (zoneMax - zoneMin || 1);
  
  // Вероятности на границах зоны
  let probAtMin = zoneProb;
  let probAtMax = zoneProb;
  
  // Если есть предыдущая зона - интерполируем с её вероятностью на границе
  if (prevZone) {
    const prevProb = Number(prevZone.metrics.avg_normalized_probability_percent);
    probAtMin = (prevProb + zoneProb) / 2;
  }
  
  // Если есть следующая зона - интерполируем с её вероятностью на границе
  if (nextZone) {
    const nextProb = Number(nextZone.metrics.avg_normalized_probability_percent);
    probAtMax = (nextProb + zoneProb) / 2;
  }
  
  // Линейная интерполяция между границами
  return probAtMin + (probAtMax - probAtMin) * positionInZone;
}

function getPriceData(price) {
  ensureDataReady();
  
  // Определяем зону на основе цены
  let foundZone = null;
  let zoneIndex = -1;
  if (state.data.zones) {
    for (let i = 0; i < state.data.zones.length; i++) {
      const zone = state.data.zones[i];
      if (price >= zone.price_range.min && price <= zone.price_range.max) {
        foundZone = zone;
        zoneIndex = i;
        break;
      }
    }
  }
  
  // Если зона найдена, используем интерполированную вероятность
  if (foundZone) {
    const prevZone = zoneIndex > 0 ? state.data.zones[zoneIndex - 1] : null;
    const nextZone = zoneIndex < state.data.zones.length - 1 ? state.data.zones[zoneIndex + 1] : null;
    
    const interpolatedProb = interpolateProbability(price, foundZone, prevZone, nextZone);
    const zoneColor = extractZoneColor(foundZone.zone_name);
    
    // Правильный расчет ожидаемой выгоды: цена * вероятность принятия
    const expectedValue = price * (interpolatedProb / 100);
    
    return {
      price,
      probability: interpolatedProb,
      expected_value: Math.max(0, expectedValue),
      zone: zoneColor,
    };
  }
  
  // Если цена вне всех зон (красная зона) - используем низкую вероятность с градиентом
  if (state.data.zones && state.data.zones.length > 0) {
    const lastZone = state.data.zones[state.data.zones.length - 1];
    if (price > lastZone.price_range.max) {
      // Чем дальше от последней зоны, тем меньше шанс
      const distanceFromZone = price - lastZone.price_range.max;
      const maxDistance = state.priceMax - lastZone.price_range.max;
      const probabilityDecay = Math.max(0, 10 - (distanceFromZone / maxDistance) * 8); // от 10% до 2%
      
      return {
        price,
        probability: probabilityDecay,
        expected_value: price * (probabilityDecay / 100),
        zone: "red",
      };
    }
    
    // Если цена меньше первой зоны
    const firstZone = state.data.zones[0];
    if (price < firstZone.price_range.min) {
      const distanceFromZone = firstZone.price_range.min - price;
      const maxDistance = firstZone.price_range.min - state.priceMin;
      const probabilityDecay = Math.max(0, 10 - (distanceFromZone / maxDistance) * 8);
      
      return {
        price,
        probability: probabilityDecay,
        expected_value: price * (probabilityDecay / 100),
        zone: "red",
      };
    }
  }
  
  // Fallback - если что-то пошло не так
  const fallback = state.data.optimal_price;
  return {
    price,
    probability: Number(fallback.normalized_probability_percent || fallback.probability_percent || 0),
    expected_value: Number(fallback.expected_value || 0),
    zone: "green",
  };
}

function updateAcceptButton() {
  const currentPrice = parseInt(priceInput.value, 10);
  if (!Number.isNaN(currentPrice)) {
    const acceptButton = document.getElementById("accept-start-price-button");
    if (acceptButton) {
      acceptButton.textContent = `Принять за ${formatCurrency(currentPrice)}`;
    }
  }
}

function updatePointerAndDisplay(price) {
  ensureDataReady();

  const boundedPrice = Math.max(state.priceMin, Math.min(state.priceMax, price));
  const positionPercent = (boundedPrice - state.priceMin) / (state.priceMax - state.priceMin || 1);

  pricePointer.style.left = `${positionPercent * 100}%`;
  priceInput.value = boundedPrice;

  const data = getPriceData(boundedPrice);
  updatePointerStyle(data.zone);

  const floatLabel = document.getElementById("pointer-float-label");
  floatLabel.textContent = `${boundedPrice}₽ (${data.probability.toFixed(2)}% P)`;

  const currentBidValueEl = document.getElementById("current-bid-value");
  if (currentBidValueEl) {
    currentBidValueEl.textContent = `${boundedPrice} ₽`;
    document.getElementById(
      "current-bid-expected-value"
    ).textContent = `Ожид. Выгода: ${data.expected_value.toFixed(2)} ₽`;
  }
  
  // Обновляем кнопку "Принять за"
  updateAcceptButton();
}


function createRecommendationsTable(data) {
  const tbody = document.getElementById("recommendations-body");
  tbody.innerHTML = "";

  // Сортируем существующие зоны
  const existingZones = (data.zones || [])
    .slice()
    .sort((a, b) => a.zone_id - b.zone_id);
  
  // Определяем, какие зоны есть
  const zoneMap = new Map();
  existingZones.forEach(zone => {
    zoneMap.set(zone.zone_id, zone);
  });
  
  // Определяем границы для пустых зон
  const lastZone = existingZones.length > 0 ? existingZones[existingZones.length - 1] : null;
  const extendedMax = lastZone ? lastZone.price_range.max + 50 : state.priceMax;
  
  // Список всех возможных зон с их метаданными
  const allPossibleZones = [
    { id: 3, name: "zone_3_green", label: "Зелёная зона", color: "green", description: "≥70% вероятность" },
    { id: 2, name: "zone_2_yellow_low", label: "Жёлтая зона (низкая)", color: "yellow", description: "50-70% вероятность" },
    { id: 1, name: "zone_1_yellow_high", label: "Жёлтая зона (высокая)", color: "yellow", description: "30-50% вероятность" },
    { id: 0, name: "zone_0_red", label: "Красная зона", color: "red", description: "<30% вероятность" }
  ];

  // Отображаем все зоны (существующие и пустые)
  allPossibleZones.forEach((zoneInfo) => {
    const tr = document.createElement("tr");
    const zoneClass = `zone-${zoneInfo.color}`;
    
    if (zoneMap.has(zoneInfo.id)) {
      // Зона существует - показываем данные
      const zone = zoneMap.get(zoneInfo.id);
      tr.innerHTML = `
        <td class="${zoneClass}">${translateZoneName(zone.zone_name)} (ID: ${zone.zone_id})</td>
        <td>${zone.price_range.min.toFixed(2)} - ${zone.price_range.max.toFixed(2)}₽</td>
        <td>${zone.metrics.avg_probability_percent.toFixed(2)}%</td>
        <td>${zone.metrics.avg_expected_value.toFixed(2)}₽</td>
        <td>${zone.metrics.avg_normalized_probability_percent.toFixed(2)}%</td>
      `;
    } else {
      // Зона отсутствует - показываем как пустую с пояснением
      const priceHint = lastZone && zoneInfo.id < 3 
        ? `>${lastZone.price_range.max.toFixed(0)}₽` 
        : "—";
      
      tr.innerHTML = `
        <td class="${zoneClass}" style="opacity: 0.5;">${zoneInfo.label} (ID: ${zoneInfo.id})</td>
        <td style="opacity: 0.5;">${priceHint}</td>
        <td style="opacity: 0.5;">—</td>
        <td style="opacity: 0.5;">—</td>
        <td style="opacity: 0.5;"><em>${zoneInfo.description}</em></td>
      `;
      tr.title = "Эта зона отсутствует в данном диапазоне цен";
    }

    tbody.appendChild(tr);
  });
}

function refreshAnalysisModal() {
  ensureDataReady();
  const optimal = state.data.optimal_price;
  const analysis = state.data.analysis;

  document.getElementById("optimal-price-value").textContent = `${optimal.price.toFixed(2)} ₽`;
  document.getElementById(
    "optimal-price-expected-value"
  ).textContent = `Ожидаемая Выгода: ${optimal.expected_value.toFixed(2)} ₽`;

  document.getElementById("optimal-price-prob").textContent = `${optimal.probability_percent.toFixed(
    2
  )}%`;

  // Определяем зону по zone_id из optimal_price
  let zoneLabel = "N/A";
  let zoneColor = "green";
  if (optimal.zone_id && state.data.zones) {
    const optimalZone = state.data.zones.find(z => z.zone_id === optimal.zone_id);
    if (optimalZone) {
      zoneLabel = translateZoneName(optimalZone.zone_name);
      zoneColor = extractZoneColor(optimalZone.zone_name);
    }
  }
  
  document.getElementById(
    "optimal-price-zone-score"
  ).innerHTML = `<strong class="zone-${zoneColor}">${zoneLabel}</strong> | ID: ${optimal.zone_id ?? "-"}`;

  document.getElementById("max-prob-price-value").textContent = `${analysis.max_probability_price.toFixed(
    0
  )} ₽`;
  document.getElementById("max-prob").textContent = `${analysis.max_probability_percent.toFixed(2)}%`;

  // Обновляем описание с актуальными данными
  const optimalPrice = optimal.price.toFixed(0);
  const optimalProb = optimal.probability_percent.toFixed(0);
  const optimalBenefit = optimal.expected_value.toFixed(0);
  const maxProbPrice = analysis.max_probability_price.toFixed(0);
  const maxProb = analysis.max_probability_percent.toFixed(0);
  
  const descriptionEl = document.getElementById("normalization-description");
  if (descriptionEl) {
    descriptionEl.innerHTML = `
      Оптимальная цена максимизирует <strong>ожидаемую выгоду</strong> = цена × вероятность принятия. 
      Например: цена ${optimalPrice}₽ с вероятностью ${optimalProb}% даёт выгоду ${optimalBenefit}₽, 
      а цена с максимальной вероятностью ${maxProbPrice}₽ (${maxProb}%) — только ${(maxProbPrice * maxProb / 100).toFixed(0)}₽.
      <br><br>
      <strong>Нормализованная вероятность</strong> показывает, насколько близка зона к максимально достижимой вероятности для этого заказа.
    `;
  }

  createRecommendationsTable(state.data);
}

function applyDataToUi(data) {
  state.data = enrichData(data);
  const bounds = computePriceBoundsFromData(state.data);
  state.priceMin = bounds.min;
  state.priceMax = bounds.max;
  state.priceStep = bounds.step;
  state.order = {
    ...state.order,
    price_start_local: Number(state.data.analysis?.start_price ?? state.order.price_start_local),
    carname: state.order.carname ?? "",
  };

  hydrateSummaryPanels(state.data);
  updateClientDetailsFromOrder();

  const initialPrice = Math.round(state.data.optimal_price.price / state.priceStep) * state.priceStep;
  updatePointerAndDisplay(initialPrice);
  state.ready = true;
  logAction("PricePilot data synced with UI");
  syncDebugControls();
}

async function bootstrap() {
  try {
    showLoading();
    const initialOrder = {
      ...state.order,
      order_timestamp: normalizeTimestamp(state.order.order_timestamp),
    };
    state.order = initialOrder;
    const data = await requestPricing(initialOrder);
    applyDataToUi(data);
  } catch (error) {
    console.error(error);
    alert("Не удалось загрузить данные PricePilot. Проверь авторизацию и API.");
  } finally {
    hideLoading();
  }
}

function handlePointerMove(clientX) {
  if (!state.ready) {
    return;
  }
  const scaleRect = priceScale.getBoundingClientRect();
  let newPointerX = clientX - scaleRect.left;

  newPointerX = Math.max(0, Math.min(scaleRect.width, newPointerX));

  const positionPercent = newPointerX / (scaleRect.width || 1);
  let calculatedPrice = state.priceMin + (state.priceMax - state.priceMin) * positionPercent;
  calculatedPrice = Math.round(calculatedPrice / state.priceStep) * state.priceStep;
  calculatedPrice = Math.max(state.priceMin, Math.min(state.priceMax, calculatedPrice));

  updatePointerAndDisplay(calculatedPrice);
}

function bindPointerEvents() {
  pricePointer.addEventListener("mousedown", (event) => {
    if (!state.ready) return;
    isDragging = true;
    pricePointer.classList.add("dragging");
    event.preventDefault();
  });

  document.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      pricePointer.classList.remove("dragging");
      const price = parseInt(priceInput.value, 10);
      logAction(`Price Bid set manually via slider to: ${price} ₽`);
    }
  });

  document.addEventListener("mousemove", (event) => {
    if (!isDragging) return;
    handlePointerMove(event.clientX);
  });

  pricePointer.addEventListener(
    "touchstart",
    (event) => {
      if (!state.ready) return;
      isDragging = true;
      pricePointer.classList.add("dragging");
      event.preventDefault();
    },
    { passive: true }
  );

  document.addEventListener("touchend", () => {
    if (isDragging) {
      isDragging = false;
      pricePointer.classList.remove("dragging");
      const price = parseInt(priceInput.value, 10);
      logAction(`Price Bid set manually via slider (Touch) to: ${price} ₽`);
    }
  });

  document.addEventListener("touchmove", (event) => {
    if (!isDragging || !event.touches[0]) return;
    handlePointerMove(event.touches[0].clientX);
  });
}

function bindInputEvents() {
  priceInput.addEventListener("change", async () => {
    if (!state.ready) return;
    let price = parseInt(priceInput.value, 10);
    if (Number.isNaN(price)) {
      price = state.data.optimal_price.price;
    }

    price = Math.round(price / state.priceStep) * state.priceStep;
    price = Math.max(state.priceMin, Math.min(state.priceMax, price));

    updatePointerAndDisplay(price);
    logAction(`Price Bid changed via manual input to: ${price} ₽`);
    await activeBidUpdate(price);
  });
}

function setOptimalPrice() {
  if (!state.ready || !state.optimalPrice) return;
  
  updatePointerAndDisplay(state.optimalPrice);
  logAction(`Optimal price set: ${state.optimalPrice}₽`);
}

async function acceptCurrentPrice() {
  try {
    ensureDataReady();
    const currentPrice = parseInt(priceInput.value, 10);
    
    if (Number.isNaN(currentPrice)) {
      alert("Пожалуйста, укажите цену.");
      return;
    }
    
    // Получаем интерполированные данные для текущей цены
    const priceData = getPriceData(currentPrice);
    const chance = Number(priceData.probability) / 100;
    
    const roll = Math.random();
    const accepted = roll <= chance;
    
    // Показываем результат симуляции
    const acceptButton = document.getElementById("accept-start-price-button");
    const originalText = acceptButton.textContent;
    const originalColor = acceptButton.style.backgroundColor;
    
    // Находим название зоны для лога
    let zoneName = "red_zone";
    if (state.data.zones) {
      for (const zone of state.data.zones) {
        if (currentPrice >= zone.price_range.min && currentPrice <= zone.price_range.max) {
          zoneName = zone.zone_name;
          break;
        }
      }
    }
    
    if (accepted) {
      acceptButton.textContent = "✓ Клиент ПРИНЯЛ!";
      acceptButton.style.backgroundColor = "var(--drivee-green)";
      logAction(`Virtual client ACCEPTED ${currentPrice}₽ (zone: ${zoneName}, prob: ${(chance * 100).toFixed(1)}%, roll: ${(roll * 100).toFixed(1)}%)`);
    } else {
      acceptButton.textContent = "✗ Клиент ОТКЛОНИЛ";
      acceptButton.style.backgroundColor = "var(--danger-color)";
      logAction(`Virtual client REJECTED ${currentPrice}₽ (zone: ${zoneName}, prob: ${(chance * 100).toFixed(1)}%, roll: ${(roll * 100).toFixed(1)}%)`);
    }
    
    // Возвращаем кнопку к нормальному виду через 2 секунды
    setTimeout(() => {
      acceptButton.textContent = originalText;
      acceptButton.style.backgroundColor = originalColor;
    }, 2000);
    
  } catch (error) {
    console.error(error);
    alert(error.message);
  }
}

async function activeBidUpdate(targetPrice) {
  try {
    showLoading();
    const normalizedPrice = Number(targetPrice);
    
    // Update only necessary fields, keep all others from state.order
    state.order = {
      ...state.order,
      price_start_local: normalizedPrice,
      order_timestamp: normalizeTimestamp(state.order.order_timestamp || Date.now() / 1000),
    };
    
    // Ensure correct types for critical fields
    if (state.order.carname == null) {
      state.order.carname = "";
    }
    if (state.order.carmodel == null) {
      state.order.carmodel = "";
    }
    if (state.order.driver_reg_date == null) {
      state.order.driver_reg_date = "";
    }
    if (state.order.driver_rating != null) {
      state.order.driver_rating = parseFloat(state.order.driver_rating);
    }
    
    console.log("🚀 Sending request to API with order:", state.order);
    const freshData = await requestPricing(state.order);
    applyDataToUi(freshData);
    updatePointerAndDisplay(normalizedPrice);
    logAction(`Получены обновленные данные для ставки ${normalizedPrice} ₽`);
  } catch (error) {
    console.error(error);
    alert(`Не удалось обновить ставку: ${error.message}`);
  } finally {
    hideLoading();
  }
}


function exitOrder() {
  logAction("Exit button clicked.");
  alert("Вы вышли из экрана заказа.");
}

function toggleTheme() {
  document.body.classList.toggle("dark-theme");
  const isDark = document.body.classList.contains("dark-theme");
  document.getElementById("theme-icon").className = `fas fa-${isDark ? "moon" : "sun"}`;
  logAction(`Theme Switched to: ${isDark ? "Dark" : "Light"}`);
}

function changePrice(delta) {
  if (!state.ready) return;
  let currentPrice = parseInt(priceInput.value || 0, 10);
  if (Number.isNaN(currentPrice)) currentPrice = state.priceMin;

  let newPrice = currentPrice + delta;
  newPrice = Math.max(state.priceMin, Math.min(state.priceMax, newPrice));
  newPrice = Math.round(newPrice / state.priceStep) * state.priceStep;

  priceInput.value = newPrice;
  updatePointerAndDisplay(newPrice);
  logAction(`Price adjusted by button: ${newPrice} ₽`);
}

function openMenu() {
  toggleDebugPanel(!state.debugPanelVisible);
}

function showAnalysis() {
  if (!state.ready) return;
  refreshAnalysisModal();
  document.getElementById("analysis-modal").style.display = "block";
  logAction("Detailed Analysis Modal Opened");
}

function closeModal() {
  document.getElementById("analysis-modal").style.display = "none";
  logAction("Detailed Analysis Modal Closed");
}

function showFromLocationModal() {
  document.getElementById("from-location-modal").style.display = "block";
  logAction("From Location Modal Opened");
}

function closeFromLocationModal() {
  document.getElementById("from-location-modal").style.display = "none";
  logAction("From Location Modal Closed");
}

function showToLocationModal() {
  document.getElementById("to-location-modal").style.display = "block";
  logAction("To Location Modal Opened");
}

function closeToLocationModal() {
  document.getElementById("to-location-modal").style.display = "none";
  logAction("To Location Modal Closed");
}

function wireGlobalHandlers() {
  window.openMenu = openMenu;
  window.exitOrder = exitOrder;
  window.toggleTheme = toggleTheme;
  
  // Обработчик для закрытия модальных окон при клике вне их
  window.onclick = function(event) {
    const analysisModal = document.getElementById("analysis-modal");
    const fromLocationModal = document.getElementById("from-location-modal");
    const toLocationModal = document.getElementById("to-location-modal");
    
    if (event.target === analysisModal) {
      closeModal();
    }
    if (event.target === fromLocationModal) {
      closeFromLocationModal();
    }
    if (event.target === toLocationModal) {
      closeToLocationModal();
    }
  };
  window.changePrice = changePrice;
  window.acceptCurrentPrice = acceptCurrentPrice;
  window.setOptimalPrice = setOptimalPrice;
  window.showAnalysis = showAnalysis;
  window.closeModal = closeModal;
  window.showFromLocationModal = showFromLocationModal;
  window.closeFromLocationModal = closeFromLocationModal;
  window.showToLocationModal = showToLocationModal;
  window.closeToLocationModal = closeToLocationModal;
  window.logAction = logAction;
  window.applyJsonOverride = applyJsonOverride;
  window.clearJsonEditor = clearJsonEditor;
  window.refreshToken = refreshToken;
  window.toggleDebugPanel = toggleDebugPanel;
}

function bindModalDismiss() {
  window.addEventListener("click", (event) => {
    const modal = document.getElementById("analysis-modal");
    if (event.target === modal) {
      closeModal();
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  pricePointer = document.getElementById("price-pointer");
  priceScale = document.getElementById("price-scale");
  priceInput = document.getElementById("price-bid");

  bindPointerEvents();
  bindInputEvents();
  bindModalDismiss();
  wireGlobalHandlers();

  const debugToggleButton = document.getElementById("debug-toggle-button");
  if (debugToggleButton) {
    debugToggleButton.addEventListener("click", () => toggleDebugPanel(!state.debugPanelVisible));
  }
  const debugOverlay = document.getElementById("debug-overlay");
  if (debugOverlay) {
    debugOverlay.addEventListener("click", () => toggleDebugPanel(false));
  }

  bootstrap();
});
