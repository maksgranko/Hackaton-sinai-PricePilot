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
    const max = Number(lastZone.price_range.max);
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
}

function hydrateSummaryPanels(data) {
  const optimal = data.optimal_price;
  
  // Сохраняем оптимальную цену в state для использования в setOptimalPrice
  state.optimalPrice = Math.round(optimal.price / state.priceStep) * state.priceStep;
  
  // Обновляем метки min/max цены - показываем общие границы
  if (data.zones && data.zones.length > 0) {
    const firstZone = data.zones[0];
    const lastZone = data.zones[data.zones.length - 1];
    
    document.getElementById("label-min-price").textContent = `${formatCurrency(
      firstZone.price_range.min
    )} (Мин)`;
    document.getElementById("label-avg-price").textContent = `${formatCurrency(
      (firstZone.price_range.min + lastZone.price_range.max) / 2
    )}`;
    document.getElementById("label-max-price").textContent = `${formatCurrency(
      lastZone.price_range.max
    )} (Макс)`;
  } else {
    document.getElementById("label-min-price").textContent = `${formatCurrency(
      state.priceMin
    )} (Мін)`;
    document.getElementById("label-avg-price").textContent = `${formatCurrency(
      (state.priceMin + state.priceMax) / 2
    )}`;
    document.getElementById("label-max-price").textContent = `${formatCurrency(
      state.priceMax
    )} (Макс)`;
  }

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

function renderZoneMarkers(data) {
  if (!data || !data.zones) return;
  
  const scaleEl = document.getElementById("price-scale");
  if (!scaleEl) return;
  
  // Удаляем старые маркеры зон
  const oldMarkers = scaleEl.querySelectorAll(".zone-marker");
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
    marker.title = `${zone.zone_name}: ${zone.price_range.min.toFixed(0)}-${zone.price_range.max.toFixed(0)}₽ (${zone.metrics.avg_probability_percent.toFixed(1)}%)`;
    
    scaleEl.appendChild(marker);
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
    
    // Интерполируем expected_value пропорционально
    const avgEV = Number(foundZone.metrics.avg_expected_value);
    const expectedValue = (price - state.order.price_start_local) * (interpolatedProb / 100);
    
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
  
  // Обновляем метки min/avg/max в зависимости от текущей зоны
  updatePriceLabels(boundedPrice);
  
  // Обновляем кнопку "Принять за"
  updateAcceptButton();
}

function updatePriceLabels(currentPrice) {
  if (!state.data || !state.data.zones) return;
  
  const labelMinEl = document.getElementById("label-min-price");
  const labelAvgEl = document.getElementById("label-avg-price");
  const labelMaxEl = document.getElementById("label-max-price");
  
  if (!labelMinEl || !labelAvgEl || !labelMaxEl) return;
  
  // Всегда показываем общие границы min-max
  const firstZone = state.data.zones[0];
  const lastZone = state.data.zones[state.data.zones.length - 1];
  
  labelMinEl.textContent = `${formatCurrency(firstZone.price_range.min)} (Мин)`;
  labelAvgEl.textContent = `${formatCurrency((firstZone.price_range.min + lastZone.price_range.max) / 2)}`;
  labelMaxEl.textContent = `${formatCurrency(lastZone.price_range.max)} (Макс)`;
}

function createRecommendationsTable(data) {
  const tbody = document.getElementById("recommendations-body");
  tbody.innerHTML = "";

  (data.zones || [])
    .slice()
    .sort((a, b) => a.zone_id - b.zone_id)
    .forEach((zone) => {
      const tr = document.createElement("tr");
      const zoneColor = extractZoneColor(zone.zone_name);
      const zoneClass = `zone-${zoneColor}`;

      tr.innerHTML = `
        <td class="${zoneClass}">${zone.zone_name} (ID: ${zone.zone_id})</td>
        <td>${zone.price_range.min.toFixed(2)} - ${zone.price_range.max.toFixed(2)}</td>
        <td>${zone.metrics.avg_probability_percent.toFixed(2)}%</td>
        <td>${zone.metrics.avg_expected_value.toFixed(2)}</td>
        <td>${zone.metrics.avg_normalized_probability_percent.toFixed(2)}%</td>
      `;

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
      zoneLabel = optimalZone.zone_name;
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

  // Обновляем описание нормализации с актуальными данными
  const normDesc = document.getElementById("normalization-description");
  if (normDesc) {
    normDesc.innerHTML = `Нормализованная вероятность (0-100%) показывает, насколько выбранная зона близка к
      <strong>максимально достижимой вероятности</strong> для этого заказа (${analysis.max_probability_percent.toFixed(2)}% при ${analysis.max_probability_price.toFixed(0)}₽).`;
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

function wireGlobalHandlers() {
  window.openMenu = openMenu;
  window.exitOrder = exitOrder;
  window.toggleTheme = toggleTheme;
  window.changePrice = changePrice;
  window.acceptCurrentPrice = acceptCurrentPrice;
  window.setOptimalPrice = setOptimalPrice;
  window.showAnalysis = showAnalysis;
  window.closeModal = closeModal;
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
