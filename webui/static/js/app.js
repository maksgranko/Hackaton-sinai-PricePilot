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
  carname: "",
  ...(CONFIG.orderDefaults || {}),
};

if (BASE_ORDER.price_start_local != null) {
  BASE_ORDER.price_start_local = Number(BASE_ORDER.price_start_local);
}
if (BASE_ORDER.carname == null) {
  BASE_ORDER.carname = "";
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
  lotteryEnabled: false,
  lastLotteryOutcome: null,
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

function syncDebugControls() {
  const panel = document.getElementById("debug-panel");
  if (!panel) return;
  const carSelect = document.getElementById("debug-carname");
  if (carSelect) {
    const value = state.order.carname ?? "";
    if (carSelect.value !== value) {
      carSelect.value = value;
    }
  }
  const lotteryToggle = document.getElementById("debug-lottery-toggle");
  if (lotteryToggle) {
    lotteryToggle.checked = state.lotteryEnabled;
  }
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
    if (state.order.price_start_local != null) {
      state.order.price_start_local = Number(state.order.price_start_local);
    }
    if (state.order.driver_rating != null) {
      state.order.driver_rating = parseFloat(state.order.driver_rating);
    }
    if (state.order.carname == null) {
      state.order.carname = "";
    }
    setDebugStatus("debug-json-status", "JSON overrides applied.", "success");
    logAction("JSON override applied via debugger.");
    const targetPrice =
      Number(state.order.price_start_local) || state.priceMin || BASE_ORDER.price_start_local;
    await activeBidUpdate(targetPrice);
    updateClientDetailsFromOrder();
  } catch (error) {
    console.error(error);
    setDebugStatus("debug-json-status", `Override failed: ${error.message}`, "error");
  }
}

async function handleCarSelection(event) {
  const value = event?.target?.value ?? "";
  state.order = {
    ...state.order,
    carname: value,
  };
  logAction(`Car brand set to "${value || "default"}" via debugger.`);
  const targetPrice =
    Number(state.order.price_start_local) || state.priceMin || BASE_ORDER.price_start_local;
  try {
    await activeBidUpdate(targetPrice);
  } catch (error) {
    console.error(error);
    setDebugStatus("debug-json-status", `Car update failed: ${error.message}`, "error");
  }
}

function toggleLotteryMode(enabled) {
  state.lotteryEnabled = enabled;
  const message = enabled
    ? "Lottery mode enabled. Client decision will be simulated."
    : "Lottery mode disabled.";
  setDebugStatus("debug-lottery-log", message, enabled ? "success" : "");
  logAction(`Lottery mode ${enabled ? "enabled" : "disabled"}.`);
  if (!enabled) {
    state.lastLotteryOutcome = null;
  }
}

function runLotterySimulation(response) {
  if (!state.lotteryEnabled || !response?.optimal_price) {
    return;
  }
  const chance = Number(response.optimal_price.normalized_probability_percent || 0) / 100;
  const roll = Math.random();
  const accepted = roll <= chance;
  const message = `${new Date().toLocaleTimeString()} — ${accepted ? "client ACCEPTED" : "client REJECTED"} @ ${response.optimal_price.price}₽ (chance ${(chance * 100).toFixed(1)}%, roll ${(roll * 100).toFixed(1)}%)`;
  setDebugStatus("debug-lottery-log", message, accepted ? "success" : "error");
  state.lastLotteryOutcome = { accepted, chance, roll };
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
  const analysis = data.analysis;

  document.getElementById("accept-start-price-button").textContent = `Принять за ${formatCurrency(
    analysis.start_price
  )}`;
  
  // Обновляем метки min/max цены в зависимости от зон
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
    )} (Мин)`;
    document.getElementById("label-avg-price").textContent = `${formatCurrency(
      (state.priceMin + state.priceMax) / 2
    )}`;
    document.getElementById("label-max-price").textContent = `${formatCurrency(
      state.priceMax
    )} (Макс)`;
  }

  const optimalPriceRounded = Math.round(optimal.price / state.priceStep) * state.priceStep;
  document.getElementById(
    "optimal-price-text"
  ).innerHTML = `<i class="fas fa-magic"></i>${optimalPriceRounded}₽`;
  
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
    marker.title = `${zone.zone_name.toUpperCase()}: ${zone.price_range.min.toFixed(0)}-${zone.price_range.max.toFixed(0)}₽ (${zone.metrics.avg_probability_percent.toFixed(1)}%)`;
    
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

function getPriceData(price) {
  ensureDataReady();
  
  // Определяем зону на основе цены
  let foundZone = null;
  if (state.data.zones) {
    for (const zone of state.data.zones) {
      if (price >= zone.price_range.min && price <= zone.price_range.max) {
        foundZone = zone;
        break;
      }
    }
  }
  
  // Если зона найдена, используем её данные
  if (foundZone) {
    const zoneColor = extractZoneColor(foundZone.zone_name);
    return {
      price,
      probability: Number(foundZone.metrics.avg_probability_percent),
      expected_value: Number(foundZone.metrics.avg_expected_value),
      zone: zoneColor,
    };
  }
  
  // Если цена вне всех зон или зон нет, используем оптимальную цену как fallback
  const fallback = state.data.optimal_price;
  let fallbackZone = "green";
  
  // Если цена больше максимальной зоны - красная
  if (state.data.zones && state.data.zones.length > 0) {
    const lastZone = state.data.zones[state.data.zones.length - 1];
    if (price > lastZone.price_range.max) {
      fallbackZone = "red";
    }
  }
  
  return {
    price,
    probability: Number(fallback.probability_percent || 0),
    expected_value: Number(fallback.expected_value || 0),
    zone: fallbackZone,
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

  (data.zones || [])
    .slice()
    .sort((a, b) => a.zone_id - b.zone_id)
    .forEach((zone) => {
      const tr = document.createElement("tr");
      const zoneColor = extractZoneColor(zone.zone_name);
      const zoneClass = `zone-${zoneColor}`;

      tr.innerHTML = `
        <td class="${zoneClass}">${zone.zone_name.toUpperCase()} (ID: ${zone.zone_id})</td>
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
      zoneLabel = optimalZone.zone_name.toUpperCase();
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
  runLotterySimulation(state.data);
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

  document.addEventListener("mouseup", async () => {
    if (isDragging) {
      isDragging = false;
      pricePointer.classList.remove("dragging");
      const price = parseInt(priceInput.value, 10);
      logAction(`Price Bid set manually via slider to: ${price} ₽`);
      await activeBidUpdate(price);
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

  document.addEventListener("touchend", async () => {
    if (isDragging) {
      isDragging = false;
      pricePointer.classList.remove("dragging");
      const price = parseInt(priceInput.value, 10);
      logAction(`Price Bid set manually via slider (Touch) to: ${price} ₽`);
      await activeBidUpdate(price);
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

async function acceptStartPrice() {
  try {
    ensureDataReady();
    const startPrice = Number(state.data.analysis.start_price);
    
    // Получаем данные для стартовой цены
    const priceData = getPriceData(startPrice);
    const chance = Number(priceData.probability) / 100;
    const roll = Math.random();
    const accepted = roll <= chance;
    
    // Показываем результат симуляции
    const acceptButton = document.getElementById("accept-start-price-button");
    const originalText = acceptButton.textContent;
    const originalColor = acceptButton.style.backgroundColor;
    
    if (accepted) {
      acceptButton.textContent = "✓ Клиент ПРИНЯЛ!";
      acceptButton.style.backgroundColor = "var(--drivee-green)";
      logAction(`Virtual client ACCEPTED ${startPrice}₽ (chance ${(chance * 100).toFixed(1)}%, roll ${(roll * 100).toFixed(1)}%)`);
    } else {
      acceptButton.textContent = "✗ Клиент ОТКЛОНИЛ";
      acceptButton.style.backgroundColor = "var(--danger-color)";
      logAction(`Virtual client REJECTED ${startPrice}₽ (chance ${(chance * 100).toFixed(1)}%, roll ${(roll * 100).toFixed(1)}%)`);
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
    state.order = {
      ...state.order,
      price_start_local: normalizedPrice,
      order_timestamp: Math.floor(Date.now() / 1000),
      carname: state.order.carname ?? "",
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

async function changePrice(delta) {
  if (!state.ready) return;
  let currentPrice = parseInt(priceInput.value || 0, 10);
  if (Number.isNaN(currentPrice)) currentPrice = state.priceMin;

  let newPrice = currentPrice + delta;
  newPrice = Math.max(state.priceMin, Math.min(state.priceMax, newPrice));
  newPrice = Math.round(newPrice / state.priceStep) * state.priceStep;

  priceInput.value = newPrice;
  updatePointerAndDisplay(newPrice);
  logAction(`Price adjusted by button: ${newPrice} ₽`);
  await activeBidUpdate(newPrice);
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
  window.acceptStartPrice = acceptStartPrice;
  window.showAnalysis = showAnalysis;
  window.closeModal = closeModal;
  window.logAction = logAction;
  window.applyJsonOverride = applyJsonOverride;
  window.clearJsonEditor = clearJsonEditor;
  window.toggleLotteryMode = toggleLotteryMode;
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

  const carSelect = document.getElementById("debug-carname");
  if (carSelect) {
    carSelect.value = state.order.carname ?? "";
    carSelect.addEventListener("change", handleCarSelection);
  }
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
