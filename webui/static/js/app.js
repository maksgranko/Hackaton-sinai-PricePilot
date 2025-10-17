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
  distance_in_meters: 3404,
  duration_in_seconds: 486,
  pickup_in_meters: 790,
  pickup_in_seconds: 169,
  driver_rating: 5,
  platform: "android",
  price_start_local: 180,
  ...(CONFIG.orderDefaults || {}),
};

if (BASE_ORDER.price_start_local != null) {
  BASE_ORDER.price_start_local = Number(BASE_ORDER.price_start_local);
}

const state = {
  token: null,
  data: null,
  order: { ...BASE_ORDER },
  priceMin: 0,
  priceMax: 0,
  priceStep: 5,
  ready: false,
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

function computePriceBoundsFromData(data) {
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
  return data;
}

function hydrateSummaryPanels(data) {
  const optimal = data.optimal_price;
  const analysis = data.analysis;

  document.getElementById("accept-start-price-button").textContent = `Принять за ${formatCurrency(
    analysis.start_price
  )}`;
  document.getElementById("label-min-price").textContent = `${formatCurrency(
    state.priceMin
  )} (Мин)`;
  document.getElementById("label-avg-price").textContent = `${formatCurrency(
    (state.priceMin + state.priceMax) / 2
  )}`;
  document.getElementById("label-max-price").textContent = `${formatCurrency(
    state.priceMax
  )} (Макс)`;

  const optimalPriceRounded = Math.round(optimal.price / state.priceStep) * state.priceStep;
  document.getElementById(
    "optimal-price-text"
  ).innerHTML = `<i class="fas fa-magic"></i>${optimalPriceRounded}₽`;
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

function getPriceData(price) {
  ensureDataReady();
  const probabilities = state.data.price_probabilities || {};
  const entries = Object.entries(probabilities);

  if (!entries.length) {
    const fallback = state.data.optimal_price;
    return {
      price,
      probability: Number(fallback.probability_percent),
      expected_value: Number(fallback.expected_value),
      zone: fallback.zone || "green",
    };
  }

  let closest = entries[0];
  let minDiff = Math.abs(Number(entries[0][0]) - price);

  for (const entry of entries.slice(1)) {
    const diff = Math.abs(Number(entry[0]) - price);
    if (diff < minDiff) {
      minDiff = diff;
      closest = entry;
    }
  }

  const basePrice = Number(closest[0]);
  const baseData = closest[1];
  const diffFromBase = price - basePrice;

  const probAdjustment = diffFromBase * 0.05;
  const evAdjustment = diffFromBase * 0.4;

  return {
    price,
    probability: Math.max(0, baseData.prob + probAdjustment),
    expected_value: Math.max(0, baseData.ev + evAdjustment),
    zone: baseData.zone,
  };
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
}

function createRecommendationsTable(data) {
  const tbody = document.getElementById("recommendations-body");
  tbody.innerHTML = "";

  (data.recommendations || [])
    .slice()
    .sort((a, b) => b.score - a.score)
    .forEach((rec) => {
      const tr = document.createElement("tr");
      const zoneClass = `zone-${rec.zone}`;

      tr.innerHTML = `
        <td class="${zoneClass}">${rec.zone.toUpperCase()} (${rec.score})</td>
        <td>${rec.price_range.min.toFixed(2)} - ${rec.price_range.max.toFixed(2)}</td>
        <td>${rec.avg_probability_percent.toFixed(2)}%</td>
        <td>${rec.avg_expected_value.toFixed(2)}</td>
        <td>${rec.normalized_probability_percent.toFixed(2)}%</td>
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

  const zoneLabel = (optimal.zone || optimal.zone_name || "N/A").toUpperCase();
  const score = optimal.score ?? "-";
  document.getElementById(
    "optimal-price-zone-score"
  ).innerHTML = `<strong class="zone-${optimal.zone || "green"}">${zoneLabel}</strong> | Score: ${score}`;

  document.getElementById("max-prob-price-value").textContent = `${analysis.max_probability_price.toFixed(
    0
  )} ₽`;
  document.getElementById("max-prob").textContent = `${analysis.max_probability_percent.toFixed(2)}%`;

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
  };

  hydrateSummaryPanels(state.data);

  const initialPrice = Math.round(state.data.optimal_price.price / state.priceStep) * state.priceStep;
  updatePointerAndDisplay(initialPrice);
  state.ready = true;
  logAction("PricePilot data synced with UI");
}

async function bootstrap() {
  try {
    const initialOrder = {
      ...state.order,
      order_timestamp: Math.floor(Date.now() / 1000),
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
      logAction(`Price Bid set manually via slider to: ${priceInput.value} ₽`);
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
      logAction(`Price Bid set manually via slider (Touch) to: ${priceInput.value} ₽`);
    }
  });

  document.addEventListener("touchmove", (event) => {
    if (!isDragging || !event.touches[0]) return;
    handlePointerMove(event.touches[0].clientX);
  });
}

function bindInputEvents() {
  priceInput.addEventListener("change", () => {
    if (!state.ready) return;
    let price = parseInt(priceInput.value, 10);
    if (Number.isNaN(price)) {
      price = state.data.optimal_price.price;
    }

    price = Math.round(price / state.priceStep) * state.priceStep;
    price = Math.max(state.priceMin, Math.min(state.priceMax, price));

    updatePointerAndDisplay(price);
    logAction(`Price Bid changed via manual input to: ${price} ₽`);
  });
}

async function acceptStartPrice() {
  try {
    ensureDataReady();
    const startPrice = Number(state.data.analysis.start_price);
    activeBidUpdate(startPrice);
  } catch (error) {
    console.error(error);
    alert(error.message);
  }
}

async function activeBidUpdate(targetPrice) {
  try {
    const normalizedPrice = Number(targetPrice);
    state.order = {
      ...state.order,
      price_start_local: normalizedPrice,
      order_timestamp: Math.floor(Date.now() / 1000),
    };
    const freshData = await requestPricing(state.order);
    applyDataToUi(freshData);
    updatePointerAndDisplay(normalizedPrice);
    logAction(`Получены обновленные данные для ставки ${normalizedPrice} ₽`);
  } catch (error) {
    console.error(error);
    alert(`Не удалось обновить ставку: ${error.message}`);
  }
}

async function confirmBid() {
  if (!state.ready) {
    alert("Данные еще не загружены.");
    return;
  }

  const bidPrice = parseInt(priceInput.value, 10);
  if (Number.isNaN(bidPrice)) {
    alert("Пожалуйста, введите цену.");
    return;
  }

  logAction(`Bid Confirmed: ${bidPrice} ₽. Запрашиваем пересчет.`);
  await activeBidUpdate(bidPrice);
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
  logAction("Menu button clicked.");
  alert("Меню открыто. Здесь будут настройки и история поездок.");
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
  window.confirmBid = confirmBid;
  window.acceptStartPrice = acceptStartPrice;
  window.showAnalysis = showAnalysis;
  window.closeModal = closeModal;
  window.logAction = logAction;
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

  bootstrap();
});
