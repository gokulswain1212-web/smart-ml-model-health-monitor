import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type { FormEvent, ReactNode } from "react";
import {
  ArrowRight,
  Activity,
  AlertTriangle,
  BarChart3,
  Bell,
  BrainCircuit,
  Boxes,
  CalendarDays,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Database,
  FileText,
  Eye,
  EyeOff,
  Home,
  HeartPulse,
  LockKeyhole,
  LogOut,
  Mail,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  RefreshCw,
  Search,
  Settings,
  Shield,
  ShieldCheck,
  Sun,
  Target,
  TrendingUp,
  Waves,
  X,
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar as RechartsBar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";
import { ProductionModel2Prediction } from "./ProductionModel2Prediction";
import {
  isDevelopmentSimulationSummary,
  isHistoricalReplaySummary,
  type HistoricalReplaySummary,
  type DevelopmentSimulationSummary,
} from "./dashboardValidation";

const API =
  import.meta.env.VITE_API_URL?.trim().replace(/\/+$/, "") ||
  (import.meta.env.DEV
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "");

async function apiFetch(url: string, init?: RequestInit) {
  const response = await fetch(url, { ...init, credentials: "include" });
  if (response.status === 401 && !url.includes("/api/v1/auth/")) {
    window.dispatchEvent(new Event("smart-ml-auth-expired"));
  }
  return response;
}
const PERIODS = [
  { label: "Last 7 days", days: 7 },
  { label: "Last 30 days", days: 30 },
  { label: "Last 90 days", days: 90 },
  { label: "Last 12 months", days: 365 },
] as const;

type MetricValues = {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  sample_count: number;
  confusion_matrix: number[][];
  roc_auc: number;
  roc_curve: {
    false_positive_rate: number;
    true_positive_rate: number;
  }[];
  performance_history?: {
    period: string;
    sample_count: number;
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  }[];
};

type DriftCause = {
  type: string;
  confidence: string;
  description: string;
  evidence: Record<string, unknown>;
};

type DriftFeatureInsight = {
  feature: string;
  feature_type: string;
  psi: number | null;
  psi_level: string | null;
  ks_statistic: number | null;
  ks_p_value: number | null;
  ks_meaningful_effect: boolean;
  severity: string;
  change_summary: string;
  reference_distribution: Record<string, unknown>[] | null;
  current_distribution: Record<string, unknown>[] | null;
  root_causes: DriftCause[];
};

type DriftInsight = {
  batch_id: string;
  timestamp: string | null;
  data_source: string;
  status: string;
  thresholds: {
    psi_warning: number;
    psi_critical: number;
    ks_significance_level: number;
    ks_min_effect_size: number;
  };
  quality_summary: {
    missing_percentage: number | null;
    duplicate_percentage: number | null;
    invalid_percentage: number | null;
    total_rows: number;
  };
  drifted_features: DriftFeatureInsight[];
  root_causes: (DriftCause & { feature: string })[];
  recommendations: string[];
};

type MonitoringAlert = {
  alert_id: string;
  batch_id: string;
  timestamp: string;
  severity: "HEALTHY" | "WARNING" | "CRITICAL";
  affected_feature: string;
  detection_reason: string;
  data_source: "REAL" | "SIMULATED" | "UNKNOWN";
  recommended_action: string;
  evidence: Record<string, unknown>;
};

type MonitoringReportSummary = {
  batch_id: string;
  timestamp: string;
  status: string;
  quality_status: string;
  source: "REAL" | "SIMULATED" | "UNKNOWN";
  rows: number;
  download_url: string;
  quality_summary: {
    missing_percentage: number | null;
    duplicate_percentage: number | null;
    invalid_percentage: number | null;
    total_rows: number;
  };
  quality_warnings: string[];
  quality_critical_reasons: string[];
  drift_summary: {
    average_psi: number | null;
    drifted_features: number;
    critical_features: number;
    feature_results: {
      feature: string;
      feature_type: string;
      psi: number;
      ks_statistic: number | null;
      ks_p_value: number | null;
      ks_meaningful_effect: boolean;
      severity: string;
    }[];
  };
};

type ProductionIngestionEvent = {
  event_id: string;
  timestamp: string;
  endpoint: "observations" | "outcomes";
  status_code: number;
  category: string;
};

type Model2PredictionHistory = {
  prediction_id: string;
  batch_id: string;
  prediction_timestamp: string;
  source: "production";
  data_source_type: "real";
  model_version: string;
  predicted_degradation_risk: 0 | 1;
  degradation_probability: number;
  risk_category: "LOW" | "MEDIUM" | "HIGH";
  health_status: "HEALTHY" | "WARNING" | "CRITICAL";
};

type DashboardData = {
  model: {
    name: string;
    algorithm: string;
    training_date: string | null;
    version: string;
    status: string;
    validation_scope: string;
    dataset_row_count: number;
    dataset_period: string;
    feature_importance: { name: string; importance: number }[];
  };
  model2: {
    status: string;
    training_eligibility: string;
    reason: string | null;
    missing_requirements: string[];
    production_batch_count: number;
    genuine_production_observation_count: number;
    labeled_batch_count: number;
    eligible_labeled_observation_count: number;
    unlabeled_batch_count: number;
    labeled_observation_count: number;
    training_dataset_size: number;
    target_class_counts: Record<string, number>;
    minimum_labeled_observations: number;
    training_progress_percent: number;
    training_size: number;
    testing_size: number;
    evaluation_dataset_size: number;
    training_class_counts: Record<string, number>;
    testing_class_counts: Record<string, number>;
    prediction_history: Model2PredictionHistory[];
    latest_training_result: {
      status: string | null;
      timestamp: string | null;
      reason: string | null;
      training_performed: boolean | null;
      model_version: string | null;
      training_size: number | null;
      testing_size: number | null;
    } | null;
    model_version: string | null;
    training_timestamp: string | null;
    evaluation_metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      roc_auc: number | null;
      pr_auc: number | null;
      confusion_matrix: number[][];
      test_class_counts: Record<string, number>;
    } | null;
  };
  model2_offline: {
    status: "TRAINED" | "NOT_TRAINED";
    model_scope: "HISTORICAL_OFFLINE";
    provenance: "HISTORICAL_OFFLINE";
    model1_replay_version: string | null;
    historical_observation_count: number;
    eligible_observation_count: number;
    target_class_counts: Record<string, number>;
    training_size: number;
    testing_size: number;
    evaluation_metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      roc_auc: number | null;
      confusion_matrix: number[][];
      test_class_counts: Record<string, number>;
    } | null;
    confusion_matrix: number[][] | null;
    evaluation_predictions: {
      batch_id: string;
      timestamp: string;
      actual_degradation_risk: number;
      predicted_degradation_risk: number;
      degradation_probability: number;
    }[];
    observation_history: {
      batch_id: string;
      timestamp: string;
      sample_count: number;
      prediction_count: number;
      actual_outcome_count: number;
      psi_score: number;
      ks_statistic: number | null;
      f1_score: number | null;
      accuracy: number | null;
      precision: number | null;
      recall: number | null;
      performance_eligible: boolean;
      target: number | null;
    }[];
    missing_requirements: string[];
    reason: string | null;
    model_version: string | null;
    training_timestamp: string | null;
    target_definition: {
      rule: string;
      f1_absolute_drop_threshold: number;
      future_window_days: number;
      target_source: string;
    } | null;
    limitations?: string[];
  };
  historical_replay: HistoricalReplaySummary;
  model2_development: {
    status: "AVAILABLE" | "UNAVAILABLE";
    training_status: string;
    provenance: string;
    model_name: string | null;
    model_version: string | null;
    algorithm: string | null;
    sample_count: number;
    training_size: number;
    testing_size: number;
    evaluation_metrics: {
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
    } | null;
    reason: string | null;
  };
  model2_development_explainability: {
    status: "AVAILABLE" | "UNAVAILABLE";
    provenance: string;
    model_version: string | null;
    importance_method: string | null;
    generation_timestamp: string | null;
    causality_warning: string;
    feature_names: string[];
    features: {
      feature: string;
      importance: number;
      rank: number;
      measures: string;
      model_rationale: string;
      risk_relation: string;
    }[];
    chart_path: string | null;
    reason?: string;
  };
  development_simulation: DevelopmentSimulationSummary;
  metrics: MetricValues;
  latest_monitoring: {
    batch_id: string;
    timestamp: string;
    status: string;
    quality_status: string;
    source: string;
    rows: number;
    reasons: string[];
  } | null;
  drift: {
    features: { name: string; score: number }[];
    average_score: number | null;
    drifted_features: number;
    critical_features: number;
    threshold: number;
    warning_threshold?: number;
    insights_by_source?: {
      real: DriftInsight | null;
      synthetic_test: DriftInsight | null;
      unknown: DriftInsight | null;
    };
  };
  quality: {
    missing_percentage: number | null;
    duplicate_percentage: number | null;
    invalid_percentage: number | null;
    total_rows: number;
  };
  predictions: {
    total: number;
    today: number;
    class_counts: { not_cancelled: number; cancelled: number };
    distribution_sample_count: number;
    batch_count: number;
    source: "real" | "simulated" | "none";
    prediction_history: {
      batch_id: string;
      row_id: string;
      observation_id: string | null;
      timestamp: string;
      source: string;
      model_version: string | null;
      predicted_class: number;
      probability_positive: number | null;
    }[];
  };
  alerts: {
    alert_id: string;
    title: string;
    level: string;
    timestamp: string;
    source: string;
    batch_id: string;
  }[];
  monitoring: {
    report_count: number;
    batch_count: number;
    simulation_report_count: number;
    simulation_batch_count: number;
    simulation_record_count: number;
    period_days: number;
    source_counts: {
      real: number;
      unknown: number;
    };
  };
  operational_performance: {
    evaluated_batch_count: number;
    evaluated_record_count: number;
    pending_outcome_count: number;
    history: {
      batch_id: string;
      timestamp: string;
      model_version: string;
      sample_count: number;
      accuracy: number;
      precision: number;
      recall: number;
      f1_score: number;
      confusion_matrix: number[][];
      data_source_type: string;
    }[];
  };
  production_observations: {
    total_observations: number;
    labeled_observations: number;
    pending_outcomes: number;
    prediction_history: {
      observation_id: string;
      batch_id: string;
      timestamp: string;
      model_version: string | null;
      predicted_class: number;
      probability_positive: number | null;
      actual_outcome: number | null;
      outcome_received_at: string | null;
    }[];
  };
  system: {
    api: string;
    model_artifact: string;
    monitoring_reports: string;
    live_outcomes: string;
    application_status: string;
    model1_readiness: string;
    schema_readiness: string;
    production_ingestion_configured: boolean;
  };
};

const navigation = [
  { label: "Overview", icon: Home, target: "overview" },
  { label: "Model 1", icon: BrainCircuit, target: "hotel-model" },
  { label: "Model 2", icon: ShieldCheck, target: "model2-risk-page" },
  { label: "Data Quality", icon: Database, target: "data-quality-page" },
  { label: "Data Drift", icon: Waves, target: "data-drift-page" },
  { label: "Model Performance", icon: TrendingUp, target: "performance-page" },
  { label: "Production Data", icon: Database, target: "ingestion-page" },
  { label: "Alerts", icon: Bell, target: "alerts-page" },
  { label: "Reports", icon: FileText, target: "reports-page" },
  { label: "Batch Explorer", icon: Boxes, target: "batch-explorer-page" },
  { label: "System Health & Settings", icon: Settings, target: "settings-page" },
];

const chartColors = ["#168cff", "#d34bff", "#0ae2c1", "#ffae25"];
const READ_ALERTS_STORAGE_KEY = "smart-ml-read-alerts";
const DESKTOP_NOTIFICATIONS_STORAGE_KEY = "smart-ml-desktop-notifications";

function loadReadAlertIds() {
  try {
    const stored = window.localStorage.getItem(READ_ALERTS_STORAGE_KEY);
    if (!stored) return new Set<string>();

    const parsed: unknown = JSON.parse(stored);
    if (Array.isArray(parsed) && parsed.every((id) => typeof id === "string")) {
      return new Set(parsed);
    }

    console.error("Stored alert read-state has an invalid format.");
  } catch (error) {
    console.error("Could not load stored alert read-state.", error);
  }
  return new Set<string>();
}

function loadDesktopNotificationPreference() {
  try {
    return (
      window.localStorage.getItem(DESKTOP_NOTIFICATIONS_STORAGE_KEY) === "true" &&
      "Notification" in window &&
      Notification.permission === "granted"
    );
  } catch (error) {
    console.error("Could not load desktop notification preference.", error);
    return false;
  }
}

function formatPercent(value: number | null | undefined) {
  return value == null || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(2)}%`;
}

function formatNumber(value: number | null | undefined) {
  return value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat().format(value);
}

function productionModel2StatusLabel(model: DashboardData["model2"]) {
  if (model.status !== "NOT_TRAINED") {
    return model.status.replaceAll("_", " ");
  }
  return model.training_eligibility === "READY"
    ? "Ready for training"
    : "Awaiting production labels";
}

function formatDate(value: string | null | undefined) {
  if (!value) return "Not recorded";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Not recorded"
    : new Intl.DateTimeFormat(undefined, {
        month: "short",
        day: "numeric",
        year: "numeric",
      }).format(date);
}

function formatTimeAgo(value: string) {
  const timestamp = new Date(value).getTime();
  if (!Number.isFinite(timestamp)) return "Time unavailable";
  const elapsedMinutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60000));
  if (elapsedMinutes < 1) return "Just now";
  if (elapsedMinutes < 60) return `${elapsedMinutes} min ago`;
  const hours = Math.floor(elapsedMinutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  return `${Math.floor(hours / 24)} day ago`;
}

function displayFeatureName(name: string) {
  return name.replaceAll("_", " ");
}

function HotelIllustration() {
  return (
    <svg
      className="hotel-illustration"
      viewBox="0 0 760 235"
      role="img"
      aria-label="Illustrated hotel beside the mountains at dusk"
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        <linearGradient id="sky" x1="0" x2="0.85" y1="0" y2="1">
          <stop offset="0" stopColor="#16184f" />
          <stop offset=".47" stopColor="#a24b91" />
          <stop offset="1" stopColor="#f19b67" />
        </linearGradient>
        <linearGradient id="glass" x1="0" x2="1" y1="0" y2="1">
          <stop stopColor="#203766" />
          <stop offset="1" stopColor="#101a39" />
        </linearGradient>
        <linearGradient id="water" x1="0" x2="0" y1="0" y2="1">
          <stop stopColor="#222868" />
          <stop offset="1" stopColor="#060d26" />
        </linearGradient>
        <linearGradient id="sunset" x1="0" x2="1" y1="0" y2="0">
          <stop stopColor="#14caff" stopOpacity=".4" />
          <stop offset=".5" stopColor="#ce52ff" stopOpacity=".17" />
          <stop offset="1" stopColor="#fb8b57" stopOpacity=".42" />
        </linearGradient>
        <filter id="windowGlow" x="-80%" y="-80%" width="260%" height="260%">
          <feGaussianBlur stdDeviation="3" />
        </filter>
      </defs>
      <rect width="760" height="235" fill="url(#sky)" />
      <circle cx="530" cy="61" r="52" fill="#ffb17a" opacity=".3" filter="url(#windowGlow)" />
      <circle cx="530" cy="61" r="23" fill="#ffd49a" opacity=".72" />
      <path d="M0 113 84 54l35 39 61-65 59 74 53-51 49 67 81-71 52 72 66-79 74 74 49-41 97 71v91H0Z" fill="#1a2149" opacity=".76" />
      <path d="m0 127 89-58 40 35 63-57 61 74 67-56 49 60 66-54 57 60 56-55 67 56 68-43 77 57v88H0Z" fill="#101a3c" opacity=".84" />
      <path d="M0 142q95-18 188 3t194-1 197 1 181-8v98H0Z" fill="url(#water)" />
      <path d="M0 151h760v36q-112-5-218 8t-242-3Q118 175 0 197Z" fill="url(#sunset)" opacity=".55" />
      <g className="hotel-building">
        <path d="m191 124 69-30 137 19v87H191Z" fill="#17284b" stroke="#f4a16d" strokeOpacity=".75" />
        <path d="m260 94 32-26 135 19v26l-130-19-37 18Z" fill="#28365c" stroke="#ffbc83" strokeOpacity=".65" />
        <path d="m397 113 30-10v91h-30Z" fill="#111a39" stroke="#f7a16d" strokeOpacity=".6" />
        <path d="m293 68 27-32 99 10v42Z" fill="#25365c" stroke="#ffbf83" strokeOpacity=".74" />
        <path d="m320 36 11-11 60 8v14Z" fill="#efb078" opacity=".8" />
        <path d="M190 125 259 95v-7l-69 29Z" fill="#faa267" opacity=".8" />
        <path d="m274 100 8-7v102h-8Z" fill="#ffb86f" opacity=".76" />
        <g fill="#ffc77b" opacity=".88">
          {Array.from({ length: 10 }, (_, index) => (
            <rect
              key={`window-a-${index}`}
              x={207 + (index % 5) * 15}
              y={132 + Math.floor(index / 5) * 27}
              width="6"
              height="11"
              rx="1"
            />
          ))}
          {Array.from({ length: 12 }, (_, index) => (
            <rect
              key={`window-b-${index}`}
              x={302 + (index % 4) * 20}
              y={91 + Math.floor(index / 4) * 24}
              width="8"
              height="12"
              rx="1"
            />
          ))}
          {Array.from({ length: 8 }, (_, index) => (
            <rect
              key={`window-c-${index}`}
              x={329 + (index % 4) * 17}
              y={47 + Math.floor(index / 4) * 21}
              width="7"
              height="11"
              rx="1"
            />
          ))}
        </g>
        <path d="M319 82h77M294 107h101M198 124h193" stroke="#bceaff" strokeOpacity=".28" />
      </g>
      <path d="M0 204q150-26 274 3t249-5 237-1" fill="none" stroke="#29baff" strokeOpacity=".75" />
      <path d="M0 215q148-22 284 2t241-5 235-2" fill="none" stroke="#a94eff" strokeOpacity=".7" />
      <path d="M0 226q171-17 300 2t232-5 228-1" fill="none" stroke="#39ebd1" strokeOpacity=".46" />
      <rect width="760" height="235" fill="url(#sunset)" opacity=".26" />
    </svg>
  );
}

function Dashboard({
  userEmail,
  onSignOut,
}: {
  userEmail: string;
  onSignOut: () => Promise<void>;
}) {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [apiError, setApiError] = useState("");
  const [connection, setConnection] = useState<"connecting" | "online" | "offline">(
    "connecting",
  );
  const [days, setDays] = useState<number>(30);
  const [darkMode, setDarkMode] = useState(true);
  const [search, setSearch] = useState("");
  const [activeSection, setActiveSection] = useState("overview");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [refreshKey, setRefreshKey] = useState(0);
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [showAllAlerts, setShowAllAlerts] = useState(false);
  const [readAlertIds, setReadAlertIds] = useState<Set<string>>(loadReadAlertIds);
  const [notificationMessage, setNotificationMessage] = useState("");
  const [desktopNotifications, setDesktopNotifications] = useState(
    loadDesktopNotificationPreference,
  );
  const [autoRefresh, setAutoRefresh] = useState(true);
  const previousAlertIds = useRef<Set<string> | null>(null);
  const notificationAreaRef = useRef<HTMLDivElement>(null);
  const [signOutError, setSignOutError] = useState("");

  async function handleSignOut() {
    setSignOutError("");
    try {
      await onSignOut();
    } catch (error) {
      setSignOutError(
        error instanceof Error ? error.message : "Could not sign out.",
      );
    }
  }

  const refresh = useCallback(() => setRefreshKey((key) => key + 1), []);
  const alertId = useCallback(
    (alert: DashboardData["alerts"][number]) =>
      alert.alert_id,
    [],
  );

  useEffect(() => {
    try {
      window.localStorage.setItem(
        READ_ALERTS_STORAGE_KEY,
        JSON.stringify([...readAlertIds]),
      );
    } catch (error) {
      console.error("Could not save alert read-state.", error);
    }
  }, [readAlertIds]);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        DESKTOP_NOTIFICATIONS_STORAGE_KEY,
        String(desktopNotifications),
      );
    } catch (error) {
      console.error("Could not save desktop notification preference.", error);
    }
  }, [desktopNotifications]);

  useEffect(() => {
    let active = true;
    const controller = new AbortController();

    async function loadDashboard() {
      try {
        const response = await apiFetch(
          `${API}/api/v1/dashboard?period_days=${days}`,
          { signal: controller.signal },
        );
        if (!response.ok) {
          throw new Error(`Dashboard API returned HTTP ${response.status}`);
        }
        const result: unknown = await response.json().catch(() => null);
        if (
          !isObject(result) ||
          !isModel2Summary(result.model2) ||
          !isOfflineModel2Summary(result.model2_offline) ||
          !isHistoricalReplaySummary(result.historical_replay)
        ) {
          throw new Error("Dashboard API returned invalid model or replay status data.");
        }
        if (!isDevelopmentSimulationSummary(result.development_simulation)) {
          throw new Error(
            "Dashboard API returned invalid development simulation data.",
          );
        }
        if (active) {
          setDashboard(result as DashboardData);
          setApiError("");
          setConnection("online");
        }
      } catch (error) {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setApiError(error instanceof Error ? error.message : "Could not load dashboard.");
          setConnection("offline");
        }
      }
    }

    void loadDashboard();
    const interval = autoRefresh
      ? window.setInterval(refresh, 30000)
      : undefined;
    return () => {
      active = false;
      controller.abort();
      if (interval !== undefined) window.clearInterval(interval);
    };
  }, [autoRefresh, days, refresh, refreshKey]);

  useEffect(() => {
    if (!dashboard) return;

    const currentIds = new Set(dashboard.alerts.map(alertId));
    const previousIds = previousAlertIds.current;
    const newAlert = previousIds
      ? dashboard.alerts.find((alert) => !previousIds.has(alertId(alert)))
      : undefined;

    if (newAlert) {
      setNotificationMessage(newAlert.title);
      if (
        desktopNotifications &&
        "Notification" in window &&
        Notification.permission === "granted"
      ) {
        new Notification("Smart ML Model Health Monitor", {
          body: newAlert.title,
        });
      }
    }

    previousAlertIds.current = currentIds;
  }, [alertId, dashboard, desktopNotifications]);

  useEffect(() => {
    if (!notificationMessage) return;
    const timeout = window.setTimeout(() => setNotificationMessage(""), 6000);
    return () => window.clearTimeout(timeout);
  }, [notificationMessage]);

  useEffect(() => {
    if (!notificationsOpen) return;

    function closeOnOutsideClick(event: PointerEvent) {
      if (
        event.target instanceof Node &&
        !notificationAreaRef.current?.contains(event.target)
      ) {
        setNotificationsOpen(false);
      }
    }

    document.addEventListener("pointerdown", closeOnOutsideClick);
    return () => document.removeEventListener("pointerdown", closeOnOutsideClick);
  }, [notificationsOpen]);

  useEffect(() => {
    function handleShortcut(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setNotificationsOpen(false);
        setSidebarOpen(false);
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        document.getElementById("dashboard-search")?.focus();
      }
    }
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [refreshKey]);

  const filteredNavigation = useMemo(
    () =>
      navigation.filter((item) =>
        item.label.toLowerCase().includes(search.trim().toLowerCase()),
      ),
    [search],
  );
  const latest = dashboard?.latest_monitoring;
  const unreadAlerts =
    dashboard?.alerts.filter((alert) => !readAlertIds.has(alertId(alert))) ?? [];
  const qualityItems = [
    {
      label: "Missing values",
      value: dashboard?.quality.missing_percentage ?? (
        dashboard?.drift.insights_by_source?.synthetic_test?.quality_summary.missing_percentage ?? null
      ),
      color: "#12d8e6",
    },
    {
      label: "Duplicate records",
      value: dashboard?.quality.duplicate_percentage ?? (
        dashboard?.drift.insights_by_source?.synthetic_test?.quality_summary.duplicate_percentage ?? null
      ),
      color: "#168cff",
    },
    {
      label: "Invalid values",
      value: dashboard?.quality.invalid_percentage ?? (
        dashboard?.drift.insights_by_source?.synthetic_test?.quality_summary.invalid_percentage ?? null
      ),
      color: "#ff9c27",
    },
  ];
  const distribution = dashboard
    ? [
        {
          name: "Not canceled",
          value: dashboard.predictions.class_counts.not_cancelled,
        },
        {
          name: "Canceled",
          value: dashboard.predictions.class_counts.cancelled,
        },
      ].filter((item) => item.value > 0)
    : [];
  const chartMetrics = dashboard
    ? [
        { name: "Accuracy", score: dashboard.metrics.accuracy },
        { name: "Precision", score: dashboard.metrics.precision },
        { name: "Recall", score: dashboard.metrics.recall },
        { name: "F1 score", score: dashboard.metrics.f1_score },
      ]
    : [];
  const confusion = dashboard?.metrics.confusion_matrix ?? [];
  const confusionTotal = confusion.flat().reduce((sum, value) => sum + value, 0);

  function navigate(target: string) {
    setActiveSection(target);
    setSidebarOpen(false);
    setNotificationsOpen(false);
  }

  function markAllAlertsRead() {
    setReadAlertIds((current) => {
      const updated = new Set(current);
      dashboard?.alerts.forEach((alert) => updated.add(alertId(alert)));
      return updated;
    });
  }

  async function enableDesktopNotifications() {
    if (!("Notification" in window)) {
      setNotificationMessage("This browser does not support desktop notifications.");
      return;
    }

    try {
      const permission = await Notification.requestPermission();
      if (permission === "granted") {
        setDesktopNotifications(true);
        setNotificationMessage("Desktop notifications are enabled.");
      } else {
        setDesktopNotifications(false);
        setNotificationMessage("Desktop notification permission was not granted.");
      }
    } catch (error) {
      console.error("Could not request desktop notification permission.", error);
      setNotificationMessage("Could not enable desktop notifications.");
    }
  }

  return (
    <div className={`app-shell ${darkMode ? "dark" : "light"} ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      {sidebarOpen && (
        <button
          type="button"
          className="sidebar-backdrop"
          aria-label="Close navigation"
          onClick={() => setSidebarOpen(false)}
        />
      )}
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand">
          <button className="brand-home" onClick={() => navigate("overview")}>
            <span className="brand-mark">
              <BrainCircuit size={29} strokeWidth={1.8} />
            </span>
            <span className="brand-name">
              <strong>Smart ML</strong>
              <small>Model Health Monitor</small>
            </span>
          </button>
          <button
            type="button"
            className="icon-button sidebar-collapse-toggle"
            onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          </button>
          <button
            type="button"
            className="icon-button mobile-close"
            onClick={() => {
              setSidebarOpen(false);
            }}
            aria-label="Close navigation"
          >
            <X size={18} />
          </button>
        </div>

        <div className="sidebar-caption">WORKSPACE</div>
        <nav className="navigation" aria-label="Dashboard sections">
          {filteredNavigation.map(({ label, icon: Icon, target }) => (
            <button
              key={target}
              className={`nav-item ${activeSection === target ? "active" : ""}`}
              onClick={() => navigate(target)}
              aria-current={activeSection === target ? "page" : undefined}
              title={label}
            >
              <Icon size={17} />
              <span>{label}</span>
              {target === "alerts-page" && unreadAlerts.length > 0 && (
                <b className="nav-count">{unreadAlerts.length}</b>
              )}
            </button>
          ))}
          {filteredNavigation.length === 0 && (
            <span className="empty-search">No matching sections</span>
          )}
        </nav>

        <div className="sidebar-art" aria-hidden="true">
          <HotelIllustration />
        </div>
        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className={`status-pulse ${connection}`} />
            <span>
              <strong>System status</strong>
              <small>
                {connection === "online"
                  ? "Connected to API"
                  : connection === "offline"
                    ? "API unavailable"
                    : "Connecting…"}
              </small>
            </span>
            <span className={`status-label ${connection}`}>
              {connection === "online" ? "Online" : connection === "offline" ? "Offline" : "…"}
            </span>
          </div>
          <div className="sidebar-tagline">
            <span className="tagline-symbol"><Activity size={19} /></span>
            <span><strong>Better models.</strong><br />Safer decisions.</span>
            <ChevronRight size={17} />
          </div>
          <span className="sidebar-version">
            Hotel Booking Model · v{dashboard?.model.version ?? "1.0.0"}
          </span>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation"
          >
            <PanelLeftOpen size={19} />
          </button>
          <label className="search-box">
            <Search size={16} />
            <input
              id="dashboard-search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search models, features, alerts..."
              aria-label="Search dashboard sections"
            />
            <kbd>⌘ K</kbd>
          </label>
          <label className="topbar-period-control">
            <CalendarDays size={15} />
            <select
              value={days}
              onChange={(event) => setDays(Number(event.target.value))}
              aria-label="Global monitoring date range"
            >
              {PERIODS.map((period) => (
                <option key={period.days} value={period.days}>{period.label}</option>
              ))}
            </select>
            <ChevronDown size={13} />
          </label>
          <button
            type="button"
            className="icon-button refresh-button"
            onClick={refresh}
            aria-label="Refresh dashboard data"
            title="Refresh data"
          >
            <RefreshCw size={16} />
          </button>
          <span className="topbar-spacer" />
          <button
            className="theme-toggle"
            onClick={() => setDarkMode((value) => !value)}
            aria-label={`Switch to ${darkMode ? "light" : "dark"} theme`}
          >
            {darkMode ? <Moon size={14} /> : <Sun size={14} />}
            <span className="toggle-knob" />
            {darkMode ? <Sun size={14} /> : <Moon size={14} />}
          </button>
          <div className="notification-area" ref={notificationAreaRef}>
            <button
              type="button"
              className="icon-button topbar-bell"
              onClick={() => setNotificationsOpen((open) => !open)}
              aria-label={`Notifications${unreadAlerts.length ? `, ${unreadAlerts.length} unread` : ""}`}
              aria-expanded={notificationsOpen}
              aria-controls="notification-popover"
            >
              <Bell size={18} />
              {unreadAlerts.length > 0 && (
                <span className="notification-count">{unreadAlerts.length}</span>
              )}
            </button>
            {notificationsOpen && (
              <section
                className="notification-popover"
                id="notification-popover"
                aria-label="Recent notifications"
              >
                <div className="notification-heading">
                  <span>
                    <strong>Notifications</strong>
                    <small>{unreadAlerts.length} unread</small>
                  </span>
                  <button
                    type="button"
                    className="text-button"
                    onClick={markAllAlertsRead}
                    disabled={unreadAlerts.length === 0}
                  >
                    Mark all read
                  </button>
                </div>
                {dashboard?.alerts.length ? (
                  <div className="notification-list">
                    {dashboard.alerts.map((alert) => {
                      const id = alertId(alert);
                      const isRead = readAlertIds.has(id);
                      return (
                        <button
                          type="button"
                          className={`notification-item ${isRead ? "read" : "unread"}`}
                          key={id}
                          onClick={() => {
                            setReadAlertIds((current) => new Set(current).add(id));
                            navigate("alerts-page");
                          }}
                        >
                          <span className={`alert-indicator ${alert.level}`}>
                            <AlertTriangle size={13} />
                          </span>
                          <span className="alert-message">
                            <strong>{alert.title}</strong>
                            <small>{formatTimeAgo(alert.timestamp)}</small>
                          </span>
                          {!isRead && <i className="unread-dot" />}
                        </button>
                      );
                    })}
                  </div>
                ) : (
                  <div className="notification-empty">You&apos;re all caught up.</div>
                )}
              </section>
            )}
          </div>
          <div className="profile-button">
            <span className="avatar">{userEmail.charAt(0).toUpperCase()}</span>
            <span className="profile-info">
              <strong>{userEmail.split("@")[0]}</strong>
              <small>Data Scientist</small>
            </span>
            <ChevronDown size={14} />
          </div>
          <button
            type="button"
            className="icon-button sign-out-button"
            onClick={() => void handleSignOut()}
            aria-label="Sign out"
            title="Sign out"
          >
            <LogOut size={17} />
          </button>
        </header>

        <div className="dashboard-content">
          {signOutError && <p className="alert-error" role="alert">{signOutError}</p>}
          {activeSection !== "overview" && (
            <DashboardPage
              page={activeSection}
              dashboard={dashboard}
              connection={connection}
              apiError={apiError}
              days={days}
              setDays={setDays}
              onRefresh={refresh}
              refreshKey={refreshKey}
              autoRefresh={autoRefresh}
              setAutoRefresh={setAutoRefresh}
              onNavigate={navigate}
            />
          )}
          {activeSection === "overview" && (
            <>
          {connection === "offline" && (
            <div className="connection-banner" role="status">
              <CircleHelp size={17} />
              <span>
                Dashboard API is unavailable. Displayed values are from the
                last successful response and may be stale. {apiError}
              </span>
              <button onClick={refresh}>Retry</button>
            </div>
          )}

          <section className="welcome-panel" id="overview">
            <HotelIllustration />
            <div className="welcome-overlay" />
            <div className="welcome-copy">
              <span className="eyebrow">MODEL HEALTH · HOTEL BOOKING</span>
              <p className="welcome-label">Welcome back,</p>
              <h1><span>{userEmail.split("@")[0]}<span className="greeting-mark">.</span></span></h1>
              <p className="welcome-description">
                Monitor your machine learning models, detect issues early, and
                keep them reliable.
              </p>
              <div className="hero-actions">
                <span className="date-chip">
                  <CalendarDays size={14} />
                  {PERIODS.find((item) => item.days === days)?.label ?? "Monitoring period"}
                </span>
                <button className="hero-link" onClick={() => navigate("hotel-model")}>
                  Model details <ChevronRight size={15} />
                </button>
              </div>
            </div>
            <div className="welcome-model">
              <span className="model-icon"><Activity size={19} /></span>
              <span>
                <small>MODEL 1</small>
                <strong>{dashboard?.model.name ?? "Hotel Booking Model"}</strong>
                <span className="model-algorithm">{dashboard?.model.algorithm ?? "Random Forest Classifier"}</span>
                <span className={`running-pill ${dashboard?.model.status === "READY" ? "" : "unavailable"}`}>
                  <i /> {dashboard?.model.status === "READY" ? "Model ready" : connection === "online" ? "Artifact unavailable" : "Connecting"}
                </span>
              </span>
            </div>
            <div className="health-gauge">
              <div className="gauge-ring">
                <span className="gauge-content">
                  <small>BATCH MONITORING</small>
                  <Activity size={20} />
                  <span className="gauge-health-label">
                    {dashboard?.latest_monitoring ? "Latest batch" : "Awaiting data"}
                  </span>
                  <strong>{dashboard?.latest_monitoring?.status ?? "—"}</strong>
                  <span>
                    {dashboard?.latest_monitoring
                      ? `${formatNumber(dashboard.latest_monitoring.rows)} rows · ${dashboard.latest_monitoring.source.replaceAll("_", " ")}`
                      : "No monitored batch"}
                  </span>
                </span>
              </div>
            </div>
            <span className="hero-dots" aria-hidden="true">···</span>
          </section>

          {dashboard && (
            <section className="metric-card-row overview-kpis" aria-label="Live monitoring summary">
              <MetricSummary
                label="Application health"
                value={dashboard.system.application_status}
                note="Backend health and schema readiness"
                icon={<HeartPulse size={22} />}
              />
              <MetricSummary
                label="Model 1"
                value={dashboard.model.status.replaceAll("_", " ")}
                note={dashboard.model.name}
                icon={<BrainCircuit size={22} />}
              />
              <MetricSummary
                label="Model 2 PRODUCTION"
                value={productionModel2StatusLabel(dashboard.model2)}
                note={dashboard.model2.reason
                  ?? `${dashboard.model2.labeled_observation_count}/${dashboard.model2.minimum_labeled_observations} genuine labeled observations`}
                icon={<ShieldCheck size={22} />}
              />
              <MetricSummary
                label="Predictions"
                value={formatNumber(dashboard.predictions.total)}
                note={`${formatNumber(dashboard.predictions.today)} today · ${dashboard.predictions.source} source`}
                icon={<Activity size={22} />}
              />
              <MetricSummary
                label="Latest quality"
                value={latest?.quality_status ?? "No data"}
                note={latest
                  ? `${latest.batch_id} · ${formatDateTime(latest.timestamp)}`
                  : "No monitored batch"}
                icon={<Database size={22} />}
              />
              <MetricSummary
                label="Drifted features"
                value={dashboard.drift.insights_by_source?.real == null && dashboard.drift.insights_by_source?.synthetic_test
                  ? String(dashboard.drift.insights_by_source.synthetic_test.drifted_features.filter((feature) => feature.severity !== "HEALTHY").length)
                  : formatNumber(dashboard.drift.drifted_features)}
                note={dashboard.drift.insights_by_source?.real == null && dashboard.drift.insights_by_source?.synthetic_test
                  ? "DEMO · synthetic drift data"
                  : dashboard.drift.average_score == null
                    ? "No drift score available"
                    : `Average PSI ${dashboard.drift.average_score.toFixed(3)}`}
                icon={<Waves size={22} />}
              />
            </section>
          )}

          <section className="dashboard-grid" aria-label="Model monitoring dashboard">
            <article className="panel metrics-panel">
              <PanelHeading
                number="1"
                title="Key Metrics"
                right={
                  <select
                    className="panel-select"
                    value={days}
                    onChange={(event) => setDays(Number(event.target.value))}
                    aria-label="Monitoring period"
                  >
                    {PERIODS.map((item) => (
                      <option key={item.days} value={item.days}>{item.label}</option>
                    ))}
                  </select>
                }
              />
              <div className="metric-tiles">
                {chartMetrics.map((item, index) => (
                  <div className={`metric-tile metric-${index}`} key={item.name}>
                    <span className="metric-icon">
                      {index === 0 ? <Target size={17} /> : index === 1 ? <BarChart3 size={17} /> : index === 2 ? <Shield size={17} /> : <Activity size={17} />}
                    </span>
                    <span className="metric-text">
                      <small>{item.name}</small>
                      <strong>{formatPercent(item.score)}</strong>
                      <span>{dashboard?.model.validation_scope ?? "Validation data"}</span>
                    </span>
                  </div>
                ))}
              </div>
              <div className="panel-footnote">
                <span><Check size={12} /></span>
                Evaluation from the model&apos;s held-out test set
              </div>
            </article>

            <article className="panel drift-summary-panel" id="data-drift">
              <PanelHeading
                number="2"
                title="Data Drift Summary"
                right={<span className="live-batch">{latest ? "LATEST BATCH" : "NO BATCHES"}</span>}
              />
              {dashboard?.drift.features.length ? (
                <>
                  <div className="drift-wave">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={dashboard.drift.features.slice(0, 10)} margin={{ top: 4, right: 0, bottom: 0, left: 0 }}>
                        <defs>
                          <linearGradient id="drift-fill" x1="0" x2="0" y1="0" y2="1">
                            <stop offset="0%" stopColor="#00e6c5" stopOpacity=".68" />
                            <stop offset="100%" stopColor="#326bff" stopOpacity=".06" />
                          </linearGradient>
                          <linearGradient id="drift-line" x1="0" x2="1">
                            <stop offset="0%" stopColor="#168cff" />
                            <stop offset="55%" stopColor="#00e3c3" />
                            <stop offset="100%" stopColor="#d64aff" />
                          </linearGradient>
                        </defs>
                        <XAxis dataKey="name" hide />
                        <YAxis hide domain={[0, "dataMax + 0.02"]} />
                        <Tooltip
                          formatter={(value) => [Number(value).toFixed(3), "PSI"]}
                          labelFormatter={(label) => displayFeatureName(String(label))}
                          contentStyle={{ background: "#0b1730", border: "1px solid #224474", borderRadius: 9, color: "#eff7ff" }}
                        />
                        <Area dataKey="score" type="monotone" stroke="url(#drift-line)" strokeWidth={2} fill="url(#drift-fill)" />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                  <div className="drift-stats">
                    {(() => {
                      const demoInsight = dashboard.drift.insights_by_source?.real == null
                        ? dashboard.drift.insights_by_source?.synthetic_test
                        : null;
                      const demoScores = demoInsight?.drifted_features
                        .map((feature) => feature.psi)
                        .filter((score): score is number => score != null) ?? [];
                      const averagePsi = demoInsight
                        ? demoScores.length
                          ? demoScores.reduce((sum, score) => sum + score, 0) / demoScores.length
                          : null
                        : dashboard.drift.average_score;
                      return (
                        <>
                          <MiniStat
                            color="green"
                            icon={<Check size={13} />}
                            value={averagePsi?.toFixed(3) ?? "DEMO"}
                            label="Average PSI"
                          />
                          <MiniStat
                            color="gold"
                            icon={<Waves size={13} />}
                            value={demoInsight
                              ? String(demoInsight.drifted_features.filter((feature) => feature.severity !== "HEALTHY").length)
                              : String(dashboard.drift.drifted_features)}
                            label="Drifted features"
                          />
                          <MiniStat
                            color="red"
                            icon={<AlertTriangle size={13} />}
                            value={demoInsight
                              ? String(demoInsight.drifted_features.filter((feature) => feature.severity === "CRITICAL").length)
                              : String(dashboard.drift.critical_features)}
                            label="Critical features"
                          />
                        </>
                      );
                    })()}
                  </div>
                  <p className="latest-batch-note">
                    {latest ? `${latest.rows} rows · ${latest.source.replaceAll("_", " ")}` : "No monitoring batch"}
                  </p>
                </>
              ) : (
                <EmptyState text="No usable feature-drift scores in the selected period." />
              )}
            </article>

            <article className="panel distribution-panel">
              <PanelHeading
                number="3"
                title="Prediction Distribution"
                right={
                  <span className="live-batch">
                    {dashboard?.predictions.source === "simulated"
                      ? "SIMULATED"
                      : dashboard?.predictions.source === "real"
                        ? "REAL"
                        : "NO PREDICTIONS"}
                  </span>
                }
              />
              {distribution.length && dashboard?.predictions.distribution_sample_count ? (
                <div className="distribution-content">
                  <div className="prediction-donut">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius="68%" outerRadius="92%" paddingAngle={3} stroke="none">
                          {distribution.map((item, index) => <Cell key={item.name} fill={index === 0 ? "#087fff" : "#c94bff"} />)}
                        </Pie>
                        <Tooltip formatter={(value) => [formatNumber(Number(value)), "Predictions"]} contentStyle={{ background: "#0b1730", border: "1px solid #224474", borderRadius: 9 }} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div className="donut-label">
                      <strong>{formatNumber(dashboard.predictions.distribution_sample_count)}</strong>
                      <small>Saved predictions</small>
                    </div>
                  </div>
                  <div className="distribution-legend">
                    {distribution.map((item, index) => {
                      const count = dashboard.predictions.distribution_sample_count;
                      return (
                        <div className="distribution-item" key={item.name}>
                          <span className={`legend-dot ${index === 0 ? "blue" : "purple"}`} />
                          <span>{item.name}</span>
                          <strong>{formatPercent(item.value / count)}</strong>
                          <small>{formatNumber(item.value)}</small>
                        </div>
                      );
                    })}
                    <small className="distribution-caption">Historical saved batches · not today&apos;s live traffic</small>
                  </div>
                </div>
              ) : (
                <EmptyState text="No saved predictions are available for this distribution." />
              )}
            </article>

            <article className="panel performance-chart-panel" id="performance-chart">
              <PanelHeading
                number="4"
                title="Model 1 Performance"
                subtitle={dashboard ? `${formatNumber(dashboard.metrics.sample_count)} held-out test records` : undefined}
                right={<span className="chart-period">TEST SET</span>}
              />
              {chartMetrics.length ? (
                <div className="performance-chart-wrap">
                  <div className="chart-legend">
                    {chartMetrics.map((item, index) => (
                      <span key={item.name}><i style={{ background: chartColors[index] }} />{item.name}</span>
                    ))}
                  </div>
                  <ResponsiveContainer width="100%" height="82%">
                    <LineChart data={[{ name: "Held-out test", ...Object.fromEntries(chartMetrics.map((item) => [item.name, item.score])) }]} margin={{ top: 12, right: 13, bottom: 0, left: -20 }}>
                      <XAxis dataKey="name" tick={{ fill: "#91a9c5", fontSize: 10 }} axisLine={{ stroke: "#25395b" }} tickLine={false} />
                      <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(value) => Number(value).toFixed(2)} tick={{ fill: "#849bb8", fontSize: 9 }} axisLine={false} tickLine={false} />
                      <Tooltip formatter={(value) => [formatPercent(Number(value)), "Score"]} contentStyle={{ background: "#0b1730", border: "1px solid #224474", borderRadius: 9 }} />
                      {chartMetrics.map((item, index) => <Line key={item.name} dataKey={item.name} type="monotone" stroke={chartColors[index]} strokeWidth={0} dot={{ r: 5, fill: chartColors[index], strokeWidth: 0 }} activeDot={{ r: 7 }} />)}
                    </LineChart>
                  </ResponsiveContainer>
                  <p className="chart-honesty-note">Only recorded hold-out evaluation is shown; its confusion matrix includes {formatNumber(confusionTotal)} test predictions. No historical outcome trend is available yet.</p>
                </div>
              ) : <EmptyState text="Model evaluation metrics are unavailable." />}
            </article>

            <article className="panel feature-drift-panel">
              <PanelHeading
                number="5"
                title="Top Feature Drift"
                right={<span className="threshold-label">Threshold {dashboard?.drift.threshold.toFixed(2) ?? "—"}</span>}
              />
              {dashboard?.drift.features.length ? (
                <div className="feature-bars">
                  {dashboard.drift.features.slice(0, 5).map((feature) => {
                    const maxValue = Math.max(dashboard.drift.threshold * 1.25, ...dashboard.drift.features.slice(0, 5).map((item) => item.score), 0.01);
                    return (
                      <div className="feature-bar-row" key={feature.name}>
                        <span title={feature.name}>{displayFeatureName(feature.name)}</span>
                        <span className="feature-bar-track">
                          <i style={{
                            width: `${Math.min(100, (feature.score / maxValue) * 100)}%`,
                            background: feature.score >= dashboard.drift.threshold ? "linear-gradient(90deg,#ff743f,#ff3d69)" : "linear-gradient(90deg,#1488ff,#bd48ff)",
                          }} />
                        </span>
                        <strong className={feature.score >= dashboard.drift.threshold ? "above-threshold" : ""}>{feature.score.toFixed(3)}</strong>
                      </div>
                    );
                  })}
                </div>
              ) : <EmptyState text="Feature drift scores will appear after a monitoring run." />}
            </article>

            <article className="panel alerts-panel" id="alerts">
              <PanelHeading
                number="6"
                title="Recent Alerts"
                right={
                  <button
                    className="text-button"
                    onClick={() => setShowAllAlerts((show) => !show)}
                    disabled={(dashboard?.alerts.length ?? 0) <= 4}
                  >
                    {showAllAlerts ? "Show less" : "View all"} <ChevronRight size={13} />
                  </button>
                }
              />
              {dashboard?.alerts.length ? (
                <div className="alerts-list">
                  {dashboard.alerts.slice(0, showAllAlerts ? dashboard.alerts.length : 4).map((alert) => (
                    <div className="alert-row" key={alert.alert_id}>
                      <span className={`alert-indicator ${alert.level}`}><AlertTriangle size={13} /></span>
                      <span className="alert-message">
                        <strong>{alert.title}</strong>
                        <small>{formatTimeAgo(alert.timestamp)} · {alert.source.replaceAll("_", " ")}</small>
                      </span>
                      <span className={`severity-pill ${alert.level}`}>{alert.level}</span>
                    </div>
                  ))}
                </div>
              ) : <EmptyState text="No persisted alerts are available in this period." />}
            </article>

            <article className="panel quality-panel">
              <PanelHeading number="7" title="Data Quality Overview" />
              <div className="quality-rings">
                {qualityItems.map((item) => (
                  <div className="quality-ring-item" key={item.label}>
                    <div
                      className="quality-ring"
                      style={{
                        background: `conic-gradient(${item.color} ${Math.min(item.value ?? 0, 100) * 3.6}deg, #1a2c4a 0deg)`,
                      }}
                    >
                      <span><Database size={15} /></span>
                    </div>
                    <strong>{item.value == null ? "—" : `${item.value.toFixed(2)}%`}</strong>
                    <small>{item.label}</small>
                  </div>
                ))}
              </div>
              <p className="panel-footnote quality-footnote">
                {latest
                  ? `Latest ${latest.source.replaceAll("_", " ")} batch · ${formatNumber(dashboard?.quality.total_rows)} rows.`
                  : "No persisted quality report is available for this period."}
              </p>
            </article>

            <article className="panel model-info-panel" id="model-info">
              <PanelHeading
                number="8"
                title="Model Information"
                right={<span className="model-status-text">{dashboard?.model.status ?? "WAITING"}</span>}
              />
              <div className="model-card">
                <div className="hotel-thumbnail"><HotelIllustration /></div>
                <span className="model-card-copy">
                  <strong>{dashboard?.model.name ?? "Hotel Booking Model"}</strong>
                  <small>{dashboard?.model.algorithm ?? "Random Forest Classifier"}</small>
                </span>
              </div>
              <div className="model-metadata">
                <div><CalendarDays size={13} /><span><small>TRAINED</small><strong>{formatDate(dashboard?.model.training_date)}</strong></span></div>
                <div><Database size={13} /><span><small>TEST RECORDS</small><strong>{formatNumber(dashboard?.metrics.sample_count)}</strong></span></div>
                <div><Shield size={13} /><span><small>VERSION</small><strong>v{dashboard?.model.version ?? "—"}</strong></span></div>
              </div>
            </article>

            <article className="panel system-panel" id="system-health">
              <PanelHeading number="9" title="System Health" />
              <div className="system-content">
                <div className="system-checks">
                  <SystemRow label="API service" value={connection === "online" ? "Online" : connection === "offline" ? "Offline" : "Connecting"} state={connection === "online" ? "good" : connection === "offline" ? "bad" : "pending"} />
                  <SystemRow label="Model artifact" value={dashboard?.system.model_artifact === "AVAILABLE" ? "Available" : dashboard?.system.model_artifact === "MISSING" ? "Missing" : "Checking"} state={dashboard?.system.model_artifact === "AVAILABLE" ? "good" : dashboard?.system.model_artifact === "MISSING" ? "bad" : "pending"} />
                  <SystemRow label="Monitoring" value={dashboard ? `${dashboard.monitoring.batch_count} batches` : "Checking"} state={dashboard && dashboard.monitoring.batch_count > 0 ? "good" : "pending"} />
                  <SystemRow label="Live outcomes" value={dashboard?.system.live_outcomes === "AVAILABLE" ? "Collected" : dashboard?.system.live_outcomes === "AWAITING" ? "Awaiting labels" : "Checking"} state={dashboard?.system.live_outcomes === "AVAILABLE" ? "good" : "pending"} />
                  <SystemRow label="Model 2 PRODUCTION risk" value={dashboard ? productionModel2StatusLabel(dashboard.model2) : "Checking"} state={dashboard?.model2.status === "TRAINED" ? "good" : "pending"} />
                </div>
                <div className="server-art" aria-hidden="true">
                  <Database size={42} />
                  <span><Shield size={20} /></span>
                  <i /><i /><i />
                </div>
              </div>
              {dashboard?.model2.status !== "TRAINED" && dashboard && (
                <p className="outcome-note">
                  {dashboard.model2.labeled_observation_count}/
                  {dashboard.model2.minimum_labeled_observations} real labeled
                  production batches. {dashboard.model2.reason}
                </p>
              )}
            </article>
          </section>

          <section className="overview-lower-grid" aria-label="Monitoring trends and drifted features">
            <DashboardPanel
              title="Historical Model 1 performance by month"
              subtitle="Original dataset held-out rows · retrospective only, not production"
            >
              <HistoricalPerformanceChart history={dashboard?.metrics.performance_history} />
            </DashboardPanel>
            <HistoricalReplayDashboardPanel
              replay={dashboard?.historical_replay}
            />
            <DashboardPanel
              title="PRODUCTION Model 2 batch prediction"
              subtitle="Uses the trained production model and real-source monitoring batches only"
            >
              {dashboard ? (
                <ProductionModel2Prediction
                  key={`${dashboard.model2.status}:${refreshKey}`}
                  apiBaseUrl={API}
                  apiFetch={apiFetch}
                  modelStatus={dashboard.model2.status}
                  modelReason={dashboard.model2.reason}
                  eligibleObservationCount={dashboard.model2.labeled_observation_count}
                  minimumObservationCount={dashboard.model2.minimum_labeled_observations}
                />
              ) : (
                <EmptyState text="Loading model readiness…" />
              )}
            </DashboardPanel>
            <DashboardPanel
              title="DEVELOPMENT Model 2 metrics"
              subtitle="Synthetic development holdout only · not production performance"
            >
              {dashboard?.model2_development.status === "AVAILABLE" &&
              dashboard.model2_development.evaluation_metrics ? (
                <div className="detail-list">
                  <DetailRow
                    label="Training status"
                    value={dashboard.model2_development.training_status.replaceAll("_", " ")}
                  />
                  <DetailRow
                    label="Accuracy"
                    value={`${(dashboard.model2_development.evaluation_metrics.accuracy * 100).toFixed(1)}%`}
                  />
                  <DetailRow
                    label="F1-score"
                    value={`${(dashboard.model2_development.evaluation_metrics.f1_score * 100).toFixed(1)}%`}
                  />
                  <DetailRow
                    label="Synthetic train / holdout records"
                    value={`${formatNumber(dashboard.model2_development.training_size)} / ${formatNumber(dashboard.model2_development.testing_size)}`}
                  />
                </div>
              ) : (
                <EmptyState
                  text={dashboard?.model2_development.reason ?? "Development metrics are unavailable."}
                />
              )}
            </DashboardPanel>
            <DevelopmentSimulationPanel
              simulation={dashboard?.development_simulation}
              error={apiError || undefined}
            />
            <DashboardPanel title="Top drifted features" subtitle="Latest available PSI summaries">
              {dashboard?.drift.features.length ? (
                <div className="overview-drift-list">
                  {[...dashboard.drift.features].sort((a, b) => b.score - a.score).slice(0, 6).map((feature) => (
                    <div className="overview-drift-row" key={feature.name}>
                      <span>{displayFeatureName(feature.name)}</span>
                      <i><b style={{ width: `${Math.min(100, Math.max(0, feature.score / Math.max(dashboard.drift.threshold, 0.01) * 100))}%` }} /></i>
                      <strong>{feature.score.toFixed(3)}</strong>
                    </div>
                  ))}
                </div>
              ) : (
                <EmptyState text="No persisted feature drift results are available." />
              )}
            </DashboardPanel>
          </section>

          <section className="reports-anchor" id="reports" aria-label="Monitoring period details">
            <span><Activity size={15} /> Monitoring coverage</span>
            <strong>{dashboard?.monitoring.report_count ?? 0} real reports</strong>
            <span>
              {dashboard?.monitoring.simulation_report_count ?? 0} dataset-simulation reports ·{" "}
              {dashboard?.monitoring.simulation_batch_count ?? 0} dataset-simulation batches in the last {days} days
            </span>
            <button onClick={refresh} aria-label="Refresh dashboard data"><Activity size={14} /> Refresh</button>
          </section>
          <section className="settings-panel" id="settings" aria-label="Dashboard settings">
            <div className="settings-title">
              <Settings size={15} />
              <strong>Settings</strong>
            </div>
            <label className="settings-toggle">
              <span>
                <strong>Automatic refresh</strong>
                <small>Check for new model data every 30 seconds</small>
              </span>
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(event) => setAutoRefresh(event.target.checked)}
              />
            </label>
            <div className="settings-notifications">
              <span>
                <strong>Desktop notifications</strong>
                <small>
                  {desktopNotifications
                    ? "Enabled for new monitoring alerts"
                    : "Optional alerts from this browser"}
                </small>
              </span>
              <button
                type="button"
                onClick={() => {
                  if (desktopNotifications) {
                    setDesktopNotifications(false);
                    setNotificationMessage("Desktop notifications are disabled.");
                  } else {
                    void enableDesktopNotifications();
                  }
                }}
              >
                {desktopNotifications ? "Disable" : "Enable"}
              </button>
            </div>
          </section>
          <footer className="dashboard-footer">
            <span>Smart ML Model Health Monitor</span>
            <span>
              {connection === "online" ? "Connected to project API" : "API connection required"}
              {" · "}
              {autoRefresh ? "Updates every 30 seconds" : "Automatic refresh paused"}
            </span>
          </footer>
            </>
          )}
        </div>
      </main>
      {notificationMessage && (
        <div className="notification-toast" role="status" aria-live="polite">
          <Bell size={16} />
          <span>{notificationMessage}</span>
          <button
            type="button"
            className="icon-button"
            aria-label="Dismiss notification"
            onClick={() => setNotificationMessage("")}
          >
            <X size={15} />
          </button>
        </div>
      )}
    </div>
  );
}

function DashboardPage({
  page,
  dashboard,
  connection,
  apiError,
  days,
  setDays,
  onRefresh,
  refreshKey,
  autoRefresh,
  setAutoRefresh,
  onNavigate,
}: {
  page: string;
  dashboard: DashboardData | null;
  connection: "connecting" | "online" | "offline";
  apiError: string;
  days: number;
  setDays: (value: number) => void;
  onRefresh: () => void;
  refreshKey: number;
  autoRefresh: boolean;
  setAutoRefresh: (value: boolean) => void;
  onNavigate: (target: string) => void;
}) {
  const selectedPage =
    navigation.find((item) => item.target === page) ?? navigation[0];
  const PageIcon = selectedPage.icon;

  return (
    <div className="workspace-page">
      {connection === "offline" && (
        <div className="workspace-error" role="alert">
          <CircleHelp size={18} />
          <span>
            Dashboard data could not be loaded. {apiError || "Check the API connection and retry."}
          </span>
          <button type="button" onClick={onRefresh}>Retry</button>
        </div>
      )}
      <header className="workspace-heading">
        <div className="workspace-title">
          <span className="workspace-title-icon"><PageIcon size={25} /></span>
          <span>
            <h1>{selectedPage.label}</h1>
            <p>{pageSubtitle(page)}</p>
          </span>
        </div>
        <label className="period-control">
          <CalendarDays size={16} />
          <select
            value={days}
            onChange={(event) => setDays(Number(event.target.value))}
            aria-label="Dashboard date range"
          >
            {PERIODS.map((period) => (
              <option key={period.days} value={period.days}>{period.label}</option>
            ))}
          </select>
          <ChevronDown size={14} />
        </label>
      </header>
      {!dashboard && connection === "connecting" ? (
        <>
          <div className="workspace-loading" role="status">
            <span className="loading-spinner" />
            <strong>Loading live model data</strong>
            <small>Connecting to the monitoring API…</small>
          </div>
          <div className="dashboard-skeleton-grid" aria-hidden="true">
            <span /><span /><span /><span />
            <span className="skeleton-wide" /><span className="skeleton-wide" />
          </div>
        </>
      ) : dashboard ? (
        <PageContent
          page={page}
          dashboard={dashboard}
          connection={connection}
          onRefresh={onRefresh}
          refreshKey={refreshKey}
          days={days}
          autoRefresh={autoRefresh}
          setAutoRefresh={setAutoRefresh}
          onNavigate={onNavigate}
        />
      ) : (
        <div className="workspace-loading workspace-empty">
          <CircleHelp size={24} />
          <strong>Live dashboard data is unavailable</strong>
          <small>Start the backend API and retry. No sample metrics are displayed.</small>
        </div>
      )}
    </div>
  );
}

function pageSubtitle(page: string) {
  switch (page) {
    case "data-quality-page":
      return "Historical missing, duplicate and invalid-value checks from saved reports.";
    case "data-drift-page":
      return "Feature distribution changes and data quality for monitored batches.";
    case "performance-page":
      return "Live production evaluation from outcomes matched to saved Model 1 predictions.";
    case "model2-risk-page":
      return "Model 2 training readiness and genuinely recorded risk observations.";
    case "batch-explorer-page":
      return "Browse monitoring reports and submit incoming batches for analysis.";
    case "reports-page":
      return "Monitoring activity and reported alerts for the selected period.";
    case "alerts-page":
      return "Review data drift and data-quality alerts from monitoring runs.";
    case "settings-page":
      return "Backend, schema, model readiness and non-secret configuration status.";
    case "ingestion-page":
      return "Trusted production collection status and the observation-to-outcome history.";
    case "monitoring-page":
      return "Service readiness, latest batch status, and data quality checks.";
    case "hotel-model":
      return "Model 1 evaluation metrics, prediction workflow, and artifact details.";
    default:
      return "Overview, performance, and details of the hotel booking cancellation prediction model.";
  }
}

function PageContent({
  page,
  dashboard,
  connection,
  onRefresh,
  refreshKey,
  days,
  autoRefresh,
  setAutoRefresh,
  onNavigate,
}: {
  page: string;
  dashboard: DashboardData;
  connection: "connecting" | "online" | "offline";
  onRefresh: () => void;
  refreshKey: number;
  days: number;
  autoRefresh: boolean;
  setAutoRefresh: (value: boolean) => void;
  onNavigate: (target: string) => void;
}) {
  switch (page) {
    case "data-quality-page":
      return <DataQualityPage dashboard={dashboard} days={days} refreshKey={refreshKey} />;
    case "model2-risk-page":
      return <Model2RiskPage dashboard={dashboard} refreshKey={refreshKey} onNavigate={onNavigate} />;
    case "batch-explorer-page":
      return <BatchExplorerPage dashboard={dashboard} days={days} onRefresh={onRefresh} refreshKey={refreshKey} />;
    case "monitoring-page":
      return (
        <MonitoringPage
          dashboard={dashboard}
          onRefresh={onRefresh}
          refreshKey={refreshKey}
        />
      );
    case "data-drift-page":
      return <DataDriftPage dashboard={dashboard} days={days} refreshKey={refreshKey} />;
    case "performance-page":
      return <PerformancePage dashboard={dashboard} onRefresh={onRefresh} onNavigate={onNavigate} />;
    case "reports-page":
      return <ReportsPage days={days} onRefresh={onRefresh} refreshKey={refreshKey} />;
    case "alerts-page":
      return <AlertsPage dashboard={dashboard} refreshKey={refreshKey} />;
    case "settings-page":
      return (
        <SettingsPage
          dashboard={dashboard}
          autoRefresh={autoRefresh}
          setAutoRefresh={setAutoRefresh}
          days={days}
          connection={connection}
        />
      );
    case "ingestion-page":
      return <ProductionIngestionPage dashboard={dashboard} refreshKey={refreshKey} onNavigate={onNavigate} />;
    default:
      return <ModelPage dashboard={dashboard} onRefresh={onRefresh} />;
  }
}

function useMonitoringReports(days: number, source: string, refreshKey: number) {
  const [reports, setReports] = useState<MonitoringReportSummary[]>([]);
  const [error, setError] = useState("");
  const [loadedQuery, setLoadedQuery] = useState("");
  const queryKey = `${days}|${source}|${refreshKey}`;

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({
      period_days: String(days),
      limit: "1000",
    });
    if (source) params.set("source", source);
    apiFetch(`${API}/api/v1/monitoring/reports?${params}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Report history request failed (${response.status}).`);
        }
        return response.json() as Promise<{ reports: MonitoringReportSummary[] }>;
      })
      .then((result) => {
        setReports(result.reports);
        setError("");
        setLoadedQuery(queryKey);
      })
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setError(requestError instanceof Error ? requestError.message : "Could not load reports.");
        setReports([]);
        setLoadedQuery(queryKey);
      });
    return () => controller.abort();
  }, [days, queryKey, source]);

  return { reports, error, loading: loadedQuery !== queryKey };
}

function DataQualityPage({
  dashboard,
  days,
  refreshKey,
}: {
  dashboard: DashboardData;
  days: number;
  refreshKey: number;
}) {
  const quality = dashboard.quality;
  const latest = dashboard.latest_monitoring;
  const [source, setSource] = useState("SIMULATED");
  const { reports, error, loading } = useMonitoringReports(days, source, refreshKey);
  const historicalReports = [...reports].reverse();
  const latestReport = reports[0];
  const isDemoQuality = latest?.source === "synthetic_test";

  return (
    <>
      <section className="metric-card-row">
        <MetricSummary
          label="Latest batch quality"
          value={latest?.quality_status ?? "No data"}
          note={latest?.batch_id ?? "No monitoring report available"}
          icon={<ShieldCheck size={22} />}
        />
        <MetricSummary
          label="Missing values"
          value={quality.missing_percentage == null
            ? "DEMO"
            : `${quality.missing_percentage.toFixed(2)}%`}
          note={`${isDemoQuality ? "DEMO · " : ""}${formatNumber(quality.total_rows)} measured rows`}
          icon={<CircleHelp size={22} />}
        />
        <MetricSummary
          label="Duplicate records"
          value={quality.duplicate_percentage == null
            ? "DEMO"
            : `${quality.duplicate_percentage.toFixed(2)}%`}
          note={isDemoQuality ? "DEMO · synthetic quality summary" : "Latest available quality summary"}
          icon={<Boxes size={22} />}
        />
        <MetricSummary
          label="Invalid values"
          value={quality.invalid_percentage == null
            ? "DEMO"
            : `${quality.invalid_percentage.toFixed(2)}%`}
          note={isDemoQuality ? "DEMO · synthetic quality summary" : "Latest available quality summary"}
          icon={<AlertTriangle size={22} />}
        />
      </section>
      <section className="two-column-page-grid">
        <DashboardPanel
          title="Quality indicators"
          subtitle="Reported by the monitoring API; no values are inferred when unavailable"
        >
          <div className="quality-indicator-list">
            <QualityBar label="Missing values" value={quality.missing_percentage} />
            <QualityBar label="Duplicate records" value={quality.duplicate_percentage} />
            <QualityBar label="Invalid values" value={quality.invalid_percentage} />
          </div>
          <p className="data-note">
            Trend values come from saved monitoring reports. Source provenance
            is preserved and simulated data is never included in production
            Model 2 training counts.
          </p>
        </DashboardPanel>
        <DashboardPanel title="Most recent quality check" subtitle="Status is not an overall model-health judgment">
          {latest ? (
            <div className="detail-list">
              <DetailRow label="Batch" value={latest.batch_id} />
              <DetailRow label="Monitored" value={formatDateTime(latest.timestamp)} />
              <DetailRow label="Quality status" value={latest.quality_status} state={latest.quality_status.toLowerCase()} />
              <DetailRow label="Rows checked" value={formatNumber(latest.rows)} />
              <DetailRow label="Source" value={latest.source.replaceAll("_", " ")} />
            </div>
          ) : (
            <EmptyState text="No saved quality report is available for this period." />
          )}
        </DashboardPanel>
      </section>
      <DashboardPanel
        title="Historical quality trends"
        subtitle="Missing, duplicate and invalid-value rates from saved reports"
      >
        <div className="alert-filters">
          <label htmlFor="quality-report-source">Report source</label>
          <select
            id="quality-report-source"
            value={source}
            onChange={(event) => setSource(event.target.value)}
          >
            <option value="REAL">Real declaration</option>
            <option value="UNKNOWN">Unknown provenance</option>
            <option value="SIMULATED">Simulated</option>
          </select>
        </div>
        {error && <p className="alert-error" role="alert">{error}</p>}
        {loading ? (
          <EmptyState text="Loading saved quality history…" />
        ) : historicalReports.length ? (
          <>
            <div className="history-trend-chart">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historicalReports}>
                  <CartesianGrid stroke="#27334b" strokeDasharray="3 5" vertical={false} />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(value) => formatDate(String(value))}
                    tick={{ fill: "#8290aa", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    domain={[0, "auto"]}
                    tickFormatter={(value) => `${Number(value).toFixed(1)}%`}
                    tick={{ fill: "#8290aa", fontSize: 10 }}
                    axisLine={false}
                    tickLine={false}
                    width={48}
                  />
                  <Tooltip
                    labelFormatter={(value) => formatDateTime(String(value))}
                    formatter={(value, name) => [`${Number(value).toFixed(2)}%`, String(name)]}
                    contentStyle={{ background: "#111827", border: "1px solid #29364e", borderRadius: 10, color: "#e9edf7" }}
                  />
                  <Line dataKey="quality_summary.missing_percentage" name="Missing" stroke="#20d9f3" dot={false} isAnimationActive={false} />
                  <Line dataKey="quality_summary.duplicate_percentage" name="Duplicates" stroke="#8b5cf6" dot={false} isAnimationActive={false} />
                  <Line dataKey="quality_summary.invalid_percentage" name="Invalid" stroke="#f59e0b" dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            {latestReport && (
              <div className="history-warning-list">
                <strong>
                  Latest report warnings · {latestReport.batch_id} · {latestReport.source}
                </strong>
                {[...latestReport.quality_critical_reasons, ...latestReport.quality_warnings].length ? (
                  <ul>
                    {[...latestReport.quality_critical_reasons, ...latestReport.quality_warnings].map((warning, index) => (
                      <li key={`${latestReport.batch_id}-${index}`}>{warning}</li>
                    ))}
                  </ul>
                ) : (
                  <p>No quality warnings or critical reasons were recorded.</p>
                )}
              </div>
            )}
          </>
        ) : (
          <EmptyState text={`No ${source.toLowerCase()} quality reports are available for this period.`} />
        )}
      </DashboardPanel>
    </>
  );
}

export function DevelopmentSimulationPanel({
  simulation,
  error,
}: {
  simulation: DashboardData["development_simulation"] | undefined;
  error?: string;
}) {
  const training = simulation?.model2_training;
  const asPercent = (value: number | null) =>
    value === null ? "—" : `${(value * 100).toFixed(1)}%`;
  return (
    <DashboardPanel
      title="DEVELOPMENT SIMULATION · synthetic only"
      subtitle="Artificial hotel bookings exercise Model 1, drift monitoring, and a chronological Model 2 development holdout. Never production data."
    >
      {!simulation && error ? (
        <div className="report-status-banner warning" role="alert">
          <AlertTriangle size={20} />
          <span>
            <strong>Development simulation data could not be loaded</strong>
            <small>{error}</small>
          </span>
        </div>
      ) : !simulation ? (
        <EmptyState text="Loading development simulation status…" />
      ) : simulation.status !== "AVAILABLE" || !training ? (
        <div className="report-status-banner warning" role="status">
          <AlertTriangle size={20} />
          <span>
            <strong>{simulation.status.replaceAll("_", " ")}</strong>
            <small>
              {simulation.reason ??
                "No completed synthetic simulation run is available."}
            </small>
          </span>
        </div>
      ) : (
        <>
          {error && (
            <div className="report-status-banner warning" role="alert">
              <AlertTriangle size={20} />
              <span>
                <strong>Dashboard refresh failed</strong>
                <small>
                  Showing the last validated simulation result. {error}
                </small>
              </span>
            </div>
          )}
          <div className="report-status-banner warning" role="status">
            <AlertTriangle size={20} />
            <span>
              <strong>DEVELOPMENT SIMULATION · NOT REAL HOTEL DATA</strong>
              <small>
                {simulation.data_source_type} · {simulation.source} ·{" "}
                {simulation.environment} · run {simulation.run_id}
              </small>
            </span>
          </div>
          <div className="detail-list">
            <DetailRow
              label="Synthetic batches / bookings"
              value={`${formatNumber(simulation.batch_count)} / ${formatNumber(simulation.booking_count)}`}
            />
            <DetailRow
              label="Model 1 baseline / minimum F1"
              value={`${asPercent(simulation.baseline_f1_score)} / ${asPercent(simulation.minimum_batch_f1_score)}`}
            />
            <DetailRow
              label="Batches with detected drift"
              value={formatNumber(simulation.drift_detected_batch_count)}
            />
            <DetailRow
              label="Model 2 development status"
              value={training.status.replaceAll("_", " ")}
            />
            <DetailRow
              label="Chronological Model 2 train / holdout"
              value={`${formatNumber(training.training_size)} / ${formatNumber(training.testing_size)}`}
            />
            <DetailRow
              label="Model 2 holdout accuracy / F1"
              value={`${(training.evaluation_metrics.accuracy * 100).toFixed(1)}% / ${(training.evaluation_metrics.f1_score * 100).toFixed(1)}%`}
            />
            <DetailRow
              label="Train target classes"
              value={Object.entries(training.training_class_counts)
                .map(([label, count]) => `${label}: ${count}`)
                .join(" · ")}
            />
            <DetailRow
              label="Holdout target classes"
              value={Object.entries(training.testing_class_counts)
                .map(([label, count]) => `${label}: ${count}`)
                .join(" · ")}
            />
          </div>
          <p className="data-note">
            This run is tagged <code>is_synthetic=true</code>. It cannot satisfy
            production Model 2 eligibility; the genuine production gate remains
            separate and unchanged.
          </p>
        </>
      )}
    </DashboardPanel>
  );
}

export function HistoricalReplayDashboardPanel({
  replay,
}: {
  replay: DashboardData["historical_replay"] | undefined;
}) {
  const eligibility = replay?.model2_eligibility;
  const metrics = replay?.model1_metrics;
  return (
    <DashboardPanel
      title="HISTORICAL OFFLINE replay"
      subtitle="Saved historical dataset replay · separate from production and synthetic development"
    >
      {!replay ? (
        <EmptyState text="Loading historical replay status…" />
      ) : replay.status !== "AVAILABLE" || !eligibility || !metrics ? (
        <div className="report-status-banner pending" role="status">
          <Database size={20} />
          <span>
            <strong>{replay.status.replaceAll("_", " ")}</strong>
            <small>
              {replay.reason ?? "No saved historical replay is available."}
            </small>
          </span>
        </div>
      ) : (
        <>
          <div className="report-status-banner warning" role="status">
            <AlertTriangle size={20} />
            <span>
              <strong>HISTORICAL OFFLINE · NOT PRODUCTION</strong>
              <small>
                {replay.provenance} · {replay.arrival_date_range[0]} to{" "}
                {replay.arrival_date_range[1]} · run {replay.run_id}
              </small>
            </span>
          </div>
          <div className="metric-card-row">
            <MetricSummary
              label="Historical booking records"
              value={formatNumber(replay.historical_record_count)}
              note="Actual historical labels; retrospective"
              icon={<Database size={22} />}
            />
            <MetricSummary
              label="Arrival-date evaluation windows"
              value={formatNumber(replay.arrival_window_count)}
              note="Not verified booking chronology"
              icon={<CalendarDays size={22} />}
            />
            <MetricSummary
              label="Model 2 labeled offline windows"
              value={formatNumber(eligibility.eligible_observation_count)}
              note={`Chronological split ${formatNumber(eligibility.training_size_after_purge)} / ${formatNumber(eligibility.testing_size)}`}
              icon={<BarChart3 size={22} />}
            />
          </div>
          <div className="detail-list">
            <strong>Model 1 retrospective metrics</strong>
            <DetailRow label="Accuracy" value={`${(metrics.accuracy * 100).toFixed(2)}%`} />
            <DetailRow label="Precision" value={`${(metrics.precision * 100).toFixed(2)}%`} />
            <DetailRow label="Recall" value={`${(metrics.recall * 100).toFixed(2)}%`} />
            <DetailRow label="F1 score" value={`${(metrics.f1_score * 100).toFixed(2)}%`} />
          </div>
          <div className="detail-list">
            <strong>Offline Model 2 eligibility · {eligibility.status}</strong>
            <DetailRow
              label="Historical target classes"
              value={Object.entries(eligibility.target_class_counts)
                .map(([label, count]) => `${label}: ${formatNumber(count)}`)
                .join(" · ") || "No eligible target labels"}
            />
            <DetailRow
              label="Holdout target classes"
              value={Object.entries(eligibility.testing_class_counts)
                .map(([label, count]) => `${label}: ${formatNumber(count)}`)
                .join(" · ") || "No holdout class counts"}
            />
            <DetailRow
              label="Model 2 training status"
              value={replay.model2_training_status?.replaceAll("_", " ") ?? "Unavailable"}
            />
            {eligibility.reason && (
              <p className="data-note" role="status">
                {eligibility.reason}
              </p>
            )}
          </div>
          <div className="detail-list">
            <strong>Replay limitations</strong>
            {replay.limitations.map((limitation) => (
              <p className="data-note" key={limitation}>{limitation}</p>
            ))}
          </div>
        </>
      )}
    </DashboardPanel>
  );
}

function Model2RiskPage({
  dashboard,
  refreshKey,
  onNavigate,
}: {
  dashboard: DashboardData;
  refreshKey: number;
  onNavigate: (target: string) => void;
}) {
  const model = dashboard.model2;
  const development = dashboard.model2_development;
  const developmentAvailable = development.status === "AVAILABLE";
  const observations = dashboard.production_observations.prediction_history;

  return (
    <>
      <OfflineModel2DashboardPanel model={dashboard.model2_offline} />
      <section className="metric-card-row">
        <MetricSummary
          label="PRODUCTION Model 2 status"
          value={productionModel2StatusLabel(model)}
          note={`${formatNumber(model.labeled_observation_count)}/${formatNumber(model.minimum_labeled_observations)} verified labels`}
          icon={<BrainCircuit size={22} />}
        />
        <MetricSummary
          label="Genuine PRODUCTION observations"
          value={formatNumber(model.genuine_production_observation_count)}
          note="Genuine production provenance only"
          icon={<Database size={22} />}
        />
        <MetricSummary
          label="PRODUCTION training dataset"
          value={formatNumber(model.training_dataset_size)}
          note="Eligible, verified labeled observations"
          icon={<FileText size={22} />}
        />
        <MetricSummary
          label="Genuine labeled PRODUCTION observations"
          value={`${formatNumber(model.labeled_observation_count)}/${formatNumber(model.minimum_labeled_observations)}`}
          note="Minimum required observations"
          icon={<Check size={22} />}
        />
        <MetricSummary
          label="PRODUCTION unlabeled batches"
          value={formatNumber(model.unlabeled_batch_count)}
          note={`${formatNumber(dashboard.production_observations.pending_outcomes)} observations await outcomes`}
          icon={<Activity size={22} />}
        />
        <MetricSummary
          label="PRODUCTION risk model version"
          value={model.model_version ?? "Unavailable"}
          note={model.training_timestamp ? formatDateTime(model.training_timestamp) : "No trained production artifact"}
          icon={<Shield size={22} />}
        />
      </section>
      <DashboardPanel
        title="PRODUCTION training eligibility"
        subtitle={`Requires ${formatNumber(model.minimum_labeled_observations)} distinct genuine labeled production batches; synthetic and development data never count`}
      >
        <div className="detail-list">
          <DetailRow
            label="Training eligibility"
            value={model.training_eligibility === "NOT_TRAINED"
              ? productionModel2StatusLabel(model)
              : model.training_eligibility.replaceAll("_", " ")}
          />
          <DetailRow label="Genuine production observations" value={formatNumber(model.genuine_production_observation_count)} />
          <DetailRow label="Eligible labeled production observations" value={formatNumber(model.eligible_labeled_observation_count)} />
          <DetailRow label="Training dataset size" value={formatNumber(model.training_dataset_size)} />
          <DetailRow label="Minimum required observations" value={formatNumber(model.minimum_labeled_observations)} />
          <DetailRow label="Eligible labeled production batches" value={formatNumber(model.labeled_batch_count)} />
          <DetailRow label="Target class distribution" value={Object.entries(model.target_class_counts).map(([label, count]) => `${label}: ${count}`).join(" · ") || "No eligible labels"} />
          <DetailRow label="Collection progress" value={`${model.training_progress_percent.toFixed(0)}%`} />
          <DetailRow
            label="Last training attempt"
            value={model.latest_training_result
              ? formatDateTime(model.latest_training_result.timestamp)
              : "No attempt recorded"}
          />
          <DetailRow
            label="Last training result"
            value={model.latest_training_result
              ? model.latest_training_result.training_performed
                ? "Production model trained"
                : model.latest_training_result.status === "NOT_TRAINED"
                  ? productionModel2StatusLabel(model)
                  : model.latest_training_result.status?.replaceAll("_", " ") ?? "Training not performed"
              : "No training attempt"}
          />
        </div>
        <progress
          className="model2-training-progress"
          aria-label="Model 2 genuine labeled production batch collection progress"
          max={100}
          value={model.training_progress_percent}
        />
        <p className="data-note">
          {model.reason ??
            model.latest_training_result?.reason ??
            "All current production training eligibility checks pass."}
        </p>
        {model.missing_requirements.length > 0 && (
          <ul className="data-note">
            {model.missing_requirements.map((requirement) => (
              <li key={requirement}>{requirement}</li>
            ))}
          </ul>
        )}
      </DashboardPanel>
      <DashboardPanel title="PRODUCTION Model 2 evaluation" subtitle="Chronological holdout metrics from verified genuine production labels only">
        {model.status === "TRAINED" ? (
          <div className="detail-list">
            <DetailRow
              label="Training eligibility"
              value={model.training_eligibility === "NOT_TRAINED"
                ? productionModel2StatusLabel(model)
                : model.training_eligibility.replaceAll("_", " ")}
            />
            <DetailRow label="Model version" value={model.model_version ?? "—"} />
            <DetailRow label="Training timestamp" value={formatDateTime(model.training_timestamp)} />
            <DetailRow label="Training rows after leakage purge" value={formatNumber(model.training_size)} />
            <DetailRow label="Evaluation dataset size" value={formatNumber(model.evaluation_dataset_size)} />
            <DetailRow label="Accuracy" value={model.evaluation_metrics ? model.evaluation_metrics.accuracy.toFixed(3) : "—"} />
            <DetailRow label="Precision" value={model.evaluation_metrics ? model.evaluation_metrics.precision.toFixed(3) : "—"} />
            <DetailRow label="Recall" value={model.evaluation_metrics ? model.evaluation_metrics.recall.toFixed(3) : "—"} />
            <DetailRow label="F1 score" value={model.evaluation_metrics ? model.evaluation_metrics.f1_score.toFixed(3) : "—"} />
            <DetailRow label="ROC-AUC" value={model.evaluation_metrics?.roc_auc == null ? "—" : model.evaluation_metrics.roc_auc.toFixed(3)} />
            <DetailRow label="PR-AUC" value={model.evaluation_metrics?.pr_auc == null ? "—" : model.evaluation_metrics.pr_auc.toFixed(3)} />
          </div>
        ) : (
          <div className="report-status-banner warning">
            <AlertTriangle size={20} />
            <span>
              <strong>{productionModel2StatusLabel(model)}</strong>
              <small>{model.reason ?? "No production risk prediction is available until the model is trained."}</small>
            </span>
          </div>
        )}
        {model.status === "TRAINED" && model.evaluation_metrics && (
          <div className="detail-list">
            <strong>PRODUCTION confusion matrix · actual vs predicted degradation</strong>
            <ConfusionMatrix matrix={model.evaluation_metrics.confusion_matrix} />
          </div>
        )}
      </DashboardPanel>
      <Model2PredictionHistory
        history={model.prediction_history}
        onNavigate={() => onNavigate("ingestion-page")}
      />
      <DashboardPanel title="PRODUCTION observation history" subtitle="Prediction records and delayed actual outcomes">
        {observations.length ? (
          <div className="alert-history-table-wrap">
            <table className="alert-history-table">
              <thead>
                <tr><th>Observation</th><th>Batch</th><th>Timestamp</th><th>Model 1 prediction</th><th>Model 1 probability</th><th>Actual outcome</th><th>Outcome received</th></tr>
              </thead>
              <tbody>
                {observations.map((observation) => (
                  <tr key={observation.observation_id}>
                    <td>{observation.observation_id}</td>
                    <td>{observation.batch_id}</td>
                    <td>{formatDateTime(observation.timestamp)}</td>
                    <td>{observation.predicted_class === 1 ? "Canceled" : "Not canceled"}</td>
                    <td>{formatPercent(observation.probability_positive)}</td>
                    <td>{observation.actual_outcome == null ? "Pending" : observation.actual_outcome === 1 ? "Canceled" : "Not canceled"}</td>
                    <td>{formatDateTime(observation.outcome_received_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <DataWorkflowGuide
            title="Waiting for trusted production observations"
            description="This history only includes real producer records paired with Model 1 predictions."
            steps={[
              "Send a real observation to the trusted producer endpoint.",
              "Keep its returned observation ID and prediction reference.",
              "Submit the matching verified outcome when available.",
            ]}
            actionLabel="Open Production Data"
            onAction={() => onNavigate("ingestion-page")}
          />
        )}
      </DashboardPanel>
      <DashboardPanel
        title="DEVELOPMENT Model 2 evaluation"
        subtitle="Metrics are read from the separate synthetic-development evaluation artifact; they are not production performance."
      >
        {developmentAvailable && development.evaluation_metrics ? (
          <>
            <div className="report-status-banner warning">
              <AlertTriangle size={20} />
              <span>
                <strong>{development.training_status.replaceAll("_", " ")}</strong>
                <small>{development.provenance} · {development.model_name} · {development.model_version}</small>
              </span>
            </div>
            <div className="metric-card-row">
              <MetricSummary
                label="DEVELOPMENT accuracy"
                value={`${(development.evaluation_metrics.accuracy * 100).toFixed(1)}%`}
                note={`${formatNumber(development.testing_size)} synthetic holdout records`}
                icon={<Target size={22} />}
              />
              <MetricSummary
                label="DEVELOPMENT F1-score"
                value={`${(development.evaluation_metrics.f1_score * 100).toFixed(1)}%`}
                note={`Trained on ${formatNumber(development.training_size)} of ${formatNumber(development.sample_count)} synthetic records`}
                icon={<BarChart3 size={22} />}
              />
            </div>
            <p className="data-note">
              Precision {(development.evaluation_metrics.precision * 100).toFixed(1)}% ·
              {" "}recall {(development.evaluation_metrics.recall * 100).toFixed(1)}%.
              These development-only metrics are kept separate from PRODUCTION observations and Model 1 evaluation.
            </p>
          </>
        ) : (
          <div className="report-status-banner warning" role="status">
            <AlertTriangle size={20} />
            <span>
              <strong>DEVELOPMENT model unavailable</strong>
              <small>{development.reason ?? "No validated development artifact is available."}</small>
            </span>
          </div>
        )}
      </DashboardPanel>
      <DevelopmentSimulationPanel simulation={dashboard.development_simulation} />
      <Model2DevelopmentRiskAssessment
        key={`model2-risk-assessment:${development.status}`}
        featureNames={dashboard.model2_development_explainability.feature_names}
        available={developmentAvailable}
      />
      <MonitoringBatchRiskAssessment
        key={`monitoring-batch-risk-assessment:${development.status}`}
        refreshKey={refreshKey}
        available={developmentAvailable}
      />
    </>
  );
}

export function OfflineModel2DashboardPanel({
  model,
}: {
  model: DashboardData["model2_offline"];
}) {
  const history = model.observation_history.map((item) => ({
    ...item,
    period: formatDateTime(item.timestamp),
  }));
  const evaluationMetrics = model.evaluation_metrics;

  return (
    <DashboardPanel
      title="HISTORICAL OFFLINE Model 2"
      subtitle="Separate Kaggle-based temporal experiment; never part of production eligibility"
    >
      <div className={`report-status-banner ${model.status === "TRAINED" ? "warning" : "pending"}`}>
        <AlertTriangle size={20} />
        <span>
          <strong>{model.status === "NOT_TRAINED" ? "Offline model unavailable" : model.status.replaceAll("_", " ")}</strong>
          <small>{model.provenance} · {model.model_version ?? "No offline artifact"}</small>
        </span>
      </div>
      <p className="data-note">
        Historical offline model: trained and evaluated using Kaggle booking data and Model 1 offline evaluation results. Not validated using live production observations.
      </p>
      <div className="metric-card-row">
        <MetricSummary
          label="Historical monitoring observations"
          value={formatNumber(model.historical_observation_count)}
          note="Grouped by actual outcome date"
          icon={<Database size={22} />}
        />
        <MetricSummary
          label="Eligible offline target labels"
          value={formatNumber(model.eligible_observation_count)}
          note={`Target classes ${Object.entries(model.target_class_counts).map(([label, count]) => `${label}: ${count}`).join(" · ") || "unavailable"}`}
          icon={<Check size={22} />}
        />
        <MetricSummary
          label="Offline train / test rows"
          value={`${formatNumber(model.training_size)} / ${formatNumber(model.testing_size)}`}
          note={model.training_timestamp ? formatDateTime(model.training_timestamp) : "No offline training"}
          icon={<BarChart3 size={22} />}
        />
      </div>
      {model.status !== "TRAINED" ? (
        <p className="data-note">{model.reason ?? "Historical offline Model 2 has not been trained."}</p>
      ) : (
        <>
          <div className="detail-list">
            <DetailRow label="Data provenance" value={model.provenance} />
            <DetailRow label="Model 1 offline replay version" value={model.model1_replay_version ?? "Unavailable"} />
            <DetailRow label="Training eligibility" value={`${model.eligible_observation_count} eligible historical labels`} />
            <DetailRow label="Target definition" value={model.target_definition?.rule ?? "Unavailable"} />
            <DetailRow label="Model 1 evaluation" value="Historical-only refit of the saved pipeline recipe; the saved random-split artifact was excluded to avoid future-period training leakage." />
          </div>
          {evaluationMetrics && (
            <>
              <div className="detail-list">
                <strong>Chronological holdout evaluation</strong>
                <DetailRow label="Accuracy" value={evaluationMetrics.accuracy.toFixed(3)} />
                <DetailRow label="Precision" value={evaluationMetrics.precision.toFixed(3)} />
                <DetailRow label="Recall" value={evaluationMetrics.recall.toFixed(3)} />
                <DetailRow label="F1 score" value={evaluationMetrics.f1_score.toFixed(3)} />
                <DetailRow label="ROC-AUC" value={evaluationMetrics.roc_auc == null ? "—" : evaluationMetrics.roc_auc.toFixed(3)} />
              </div>
              {model.confusion_matrix && (
                <div className="detail-list">
                  <strong>Offline confusion matrix · actual vs predicted degradation</strong>
                  <ConfusionMatrix matrix={model.confusion_matrix} />
                </div>
              )}
            </>
          )}
          {history.length > 0 && (
            <section className="two-column-page-grid">
              <div>
                <h3>Historical PSI trend</h3>
                <div style={{ width: "100%", height: 260 }}>
                  <ResponsiveContainer>
                    <LineChart data={history} margin={{ top: 8, right: 20, bottom: 28, left: 8 }}>
                      <CartesianGrid stroke="#20324c" strokeDasharray="3 3" />
                      <XAxis dataKey="period" tick={{ fill: "#a9bad1", fontSize: 9 }} angle={-30} textAnchor="end" />
                      <YAxis tick={{ fill: "#a9bad1", fontSize: 10 }} />
                      <Tooltip />
                      <Line dataKey="psi_score" name="Mean PSI" stroke="#22c6a8" dot={false} connectNulls={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <h3>Historical KS trend</h3>
                <div style={{ width: "100%", height: 260 }}>
                  <ResponsiveContainer>
                    <LineChart data={history} margin={{ top: 8, right: 20, bottom: 28, left: 8 }}>
                      <CartesianGrid stroke="#20324c" strokeDasharray="3 3" />
                      <XAxis dataKey="period" tick={{ fill: "#a9bad1", fontSize: 9 }} angle={-30} textAnchor="end" />
                      <YAxis domain={[0, 1]} tick={{ fill: "#a9bad1", fontSize: 10 }} />
                      <Tooltip />
                      <Line dataKey="ks_statistic" name="Mean KS statistic" stroke="#f59e0b" dot={false} connectNulls={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <div>
                <h3>Model 1 F1 trend from actual Kaggle labels</h3>
                <div style={{ width: "100%", height: 260 }}>
                  <ResponsiveContainer>
                    <LineChart data={history} margin={{ top: 8, right: 20, bottom: 28, left: 8 }}>
                      <CartesianGrid stroke="#20324c" strokeDasharray="3 3" />
                      <XAxis dataKey="period" tick={{ fill: "#a9bad1", fontSize: 9 }} angle={-30} textAnchor="end" />
                      <YAxis domain={[0, 1]} tick={{ fill: "#a9bad1", fontSize: 10 }} />
                      <Tooltip />
                      <Line dataKey="f1_score" name="Model 1 F1" stroke="#60a5fa" dot={false} connectNulls={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </section>
          )}
          <DashboardPanel title="Offline holdout health predictions" subtitle="Predictions are shown only for the chronological offline evaluation partition">
            {model.evaluation_predictions.length > 0 ? (
              <div className="alert-history-table-wrap">
                <table className="alert-history-table">
                  <thead>
                    <tr><th>Offline batch</th><th>Outcome period</th><th>Actual degradation</th><th>Predicted degradation</th><th>Probability</th></tr>
                  </thead>
                  <tbody>
                    {model.evaluation_predictions.map((prediction) => (
                      <tr key={prediction.batch_id}>
                        <td>{prediction.batch_id}</td>
                        <td>{formatDateTime(prediction.timestamp)}</td>
                        <td>{prediction.actual_degradation_risk ? "Degraded" : "Stable"}</td>
                        <td>{prediction.predicted_degradation_risk ? "Degraded" : "Stable"}</td>
                        <td>{formatPercent(prediction.degradation_probability)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <EmptyState text="No chronological offline holdout predictions are available." />}
          </DashboardPanel>
          {model.limitations?.map((limitation) => (
            <p className="data-note" key={limitation}>{limitation}</p>
          ))}
        </>
      )}
      {model.missing_requirements.length > 0 && (
        <ul className="data-note">
          {model.missing_requirements.map((requirement) => (
            <li key={requirement}>{requirement}</li>
          ))}
        </ul>
      )}
    </DashboardPanel>
  );
}

function Model2PredictionHistory({
  history,
  onNavigate,
}: {
  history: Model2PredictionHistory[];
  onNavigate: () => void;
}) {
  const chartData = [...history].reverse();
  return (
    <DashboardPanel
      title="PRODUCTION degradation-risk history"
      subtitle="Persisted Model 2 predictions for genuine production monitoring batches only"
    >
      {history.length ? (
        <>
          <div className="history-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <CartesianGrid stroke="#25395b" strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="batch_id" tick={{ fill: "#91a9c5", fontSize: 9 }} axisLine={{ stroke: "#25395b" }} tickLine={false} minTickGap={18} />
                <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(value) => `${Math.round(Number(value) * 100)}%`} tick={{ fill: "#91a9c5", fontSize: 9 }} axisLine={false} tickLine={false} width={42} />
                <Tooltip
                  formatter={(value) => [`${(Number(value) * 100).toFixed(1)}%`, "Degradation risk"]}
                  labelFormatter={(label, payload) => `${String(label)} · ${formatDateTime(payload[0]?.payload.prediction_timestamp)}`}
                  contentStyle={{ background: "#0b1730", border: "1px solid #224474", borderRadius: 9, color: "#eff7ff" }}
                />
                <Line dataKey="degradation_probability" name="Degradation probability" stroke="#ffae25" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="alert-history-table-wrap">
            <table className="alert-history-table">
              <thead>
                <tr><th>Source</th><th>Batch</th><th>Prediction time</th><th>Predicted degradation</th><th>Risk probability</th><th>Health status</th><th>Model version</th></tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.prediction_id}>
                    <td>PRODUCTION</td>
                    <td>{item.batch_id}</td>
                    <td>{formatDateTime(item.prediction_timestamp)}</td>
                    <td>{item.predicted_degradation_risk === 1 ? "Yes" : "No"}</td>
                    <td>{formatPercent(item.degradation_probability)}</td>
                    <td>{item.health_status}</td>
                    <td>{item.model_version}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <DataWorkflowGuide
          title="Risk predictions start after production training"
          description="Model 2 stays unavailable for production scoring until its genuine-label requirements are met."
          steps={[
            "Collect at least 20 distinct genuine production batches with verified outcomes.",
            "Complete the chronological training and leakage checks.",
            "New risk scores will be saved here after the production model is trained.",
          ]}
          actionLabel="View training readiness"
          onAction={onNavigate}
        />
      )}
    </DashboardPanel>
  );
}

function ProductionIngestionPage({
  dashboard,
  refreshKey,
  onNavigate,
}: {
  dashboard: DashboardData;
  refreshKey: number;
  onNavigate: (target: string) => void;
}) {
  const model = dashboard.model2;
  const observations = dashboard.production_observations.prediction_history;
  const configured = dashboard.system.production_ingestion_configured;
  const [events, setEvents] = useState<ProductionIngestionEvent[]>([]);
  const [historyError, setHistoryError] = useState("");
  const [loadedRefreshKey, setLoadedRefreshKey] = useState<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    apiFetch(`${API}/api/v1/monitoring/ingestion-status`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Ingestion history request failed (${response.status}).`);
        }
        return response.json() as Promise<{
          events: ProductionIngestionEvent[];
        }>;
      })
      .then((result) => {
        setEvents(result.events);
        setHistoryError("");
        setLoadedRefreshKey(refreshKey);
      })
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setEvents([]);
        setHistoryError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load ingestion history.",
        );
        setLoadedRefreshKey(refreshKey);
      });
    return () => controller.abort();
  }, [refreshKey]);
  const historyLoading = loadedRefreshKey !== refreshKey;

  const eventCategoryLabel: Record<string, string> = {
    accepted: "Accepted",
    authentication_rejected: "Authentication rejected",
    observation_not_found: "Observation not found",
    duplicate_submission: "Duplicate submission",
    validation_rejected: "Validation rejected",
    ingestion_not_configured_or_unavailable: "Not configured or unavailable",
    request_rejected: "Request rejected",
  };
  const productionSummary = [
    { name: "Production batches", value: model.production_batch_count },
    { name: "Genuine observations", value: dashboard.production_observations.total_observations },
    { name: "Labeled outcomes", value: dashboard.production_observations.labeled_observations },
    { name: "Pending outcomes", value: dashboard.production_observations.pending_outcomes },
  ];
  return (
    <>
      <section className="metric-card-row production-summary-metrics">
        <MetricSummary
          label="Ingestion authentication"
          value={configured ? "Configured" : "Not configured"}
          note={configured ? "Server-side token check is enabled" : "Trusted producer ingestion is blocked"}
          icon={<LockKeyhole size={22} />}
        />
        <MetricSummary
          label="Production batches"
          value={formatNumber(model.production_batch_count)}
          note="Token-authenticated producer submissions only"
          icon={<Database size={22} />}
        />
        <MetricSummary
          label="Total production observations"
          value={formatNumber(dashboard.production_observations.total_observations)}
          note="Genuine observations received from trusted producers"
          icon={<FileText size={22} />}
        />
        <MetricSummary
          label="Verified actual outcomes"
          value={formatNumber(dashboard.production_observations.labeled_observations)}
          note="Genuine labels matched to production predictions"
          icon={<Check size={22} />}
        />
        <MetricSummary
          label="Eligible Model 2 batches"
          value={`${formatNumber(model.labeled_batch_count)}/${formatNumber(model.minimum_labeled_observations)}`}
          note="Distinct batches with a verified target"
          icon={<ShieldCheck size={22} />}
        />
        <MetricSummary
          label="Unlabeled batches"
          value={formatNumber(model.unlabeled_batch_count)}
          note={`${formatNumber(dashboard.production_observations.pending_outcomes)} observations await outcomes`}
          icon={<Activity size={22} />}
        />
      </section>
      <DashboardPanel
        title="Production Data Summary"
        subtitle="Counts from genuine token-authenticated production data"
      >
        <div style={{ width: "100%", height: 280 }}>
          <ResponsiveContainer>
            <BarChart
              data={productionSummary}
              margin={{ top: 12, right: 24, bottom: 18, left: 8 }}
            >
              <CartesianGrid stroke="#20324c" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="name"
                tick={{ fill: "#8290aa", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                allowDecimals={false}
                domain={[0, "dataMax + 1"]}
                tick={{ fill: "#8290aa", fontSize: 11 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                formatter={(value) => [formatNumber(Number(value)), "Production count"]}
                contentStyle={{
                  background: "#0b1730",
                  border: "1px solid #224474",
                  borderRadius: 9,
                  color: "#eff7ff",
                }}
              />
              <RechartsBar
                dataKey="value"
                fill="#22c6a8"
                radius={[5, 5, 0, 0]}
                isAnimationActive={false}
              >
                <LabelList
                  dataKey="value"
                  position="top"
                  fill="#dce7f5"
                  formatter={(value) => formatNumber(Number(value))}
                />
              </RechartsBar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </DashboardPanel>
      <DashboardPanel
        title="Trusted producer endpoints"
        subtitle="Production credentials stay on the producer and backend; they are never sent to this browser"
      >
        <div className={`report-status-banner ${configured ? "healthy" : "warning"}`}>
          {configured ? <ShieldCheck size={20} /> : <AlertTriangle size={20} />}
          <span>
            <strong>
              {configured
                ? "Production ingestion is configured"
                : "Production ingestion is not configured"}
            </strong>
            <small>
              {configured
                    ? "A sufficiently long producer token is configured; this does not confirm that an external hotel producer is connected. The secret is not exposed."
                : "Set PRODUCTION_INGESTION_TOKEN in the backend environment and securely configure the trusted producer."}
            </small>
          </span>
        </div>
        <div className="detail-list">
          <DetailRow label="Observation endpoint" value="POST /api/v1/monitoring/observations" />
          <DetailRow label="Outcome endpoint" value="POST /api/v1/monitoring/outcomes" />
          <DetailRow label="Authentication header" value="X-Production-Ingestion-Token" />
          <DetailRow
            label="Training eligibility"
            value={`${formatNumber(model.labeled_batch_count)} / ${formatNumber(model.minimum_labeled_observations)} labeled batches`}
          />
          <DetailRow
            label="Missing requirements"
            value={model.missing_requirements.join(" ") || "No outstanding requirements"}
          />
          <DetailRow
            label="Training status"
            value={productionModel2StatusLabel(model)}
            state={model.status === "TRAINED" ? "good" : "pending"}
          />
        </div>
        <p className="data-note">
          This dashboard does not collect trusted producer credentials or submit
          production labels. Authentication failures (401/503) and validation or
          duplicate errors (404/409/422) are summarized in a bounded in-memory
          event list. Request bodies, observation values and credentials are
          never recorded; this short event history resets when the API restarts.
        </p>
      </DashboardPanel>
      <DashboardPanel
        title="Recent ingestion events"
        subtitle="Last 50 accepted or rejected attempts · sanitized · current API process"
      >
        {historyError && <p className="alert-error" role="alert">{historyError}</p>}
        {historyLoading ? (
          <EmptyState text="Loading ingestion event history…" />
        ) : events.length ? (
          <div className="alert-history-table-wrap">
            <table className="alert-history-table">
              <thead>
                <tr><th>Endpoint</th><th>Result</th><th>HTTP</th><th>Timestamp</th></tr>
              </thead>
              <tbody>
                {events.map((event) => (
                  <tr key={event.event_id}>
                    <td>{event.endpoint}</td>
                    <td>{eventCategoryLabel[event.category] ?? "Request status recorded"}</td>
                    <td>{event.status_code}</td>
                    <td>{formatDateTime(event.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <DataWorkflowGuide
            title="No ingestion attempts in this API process"
            description="This event log records sanitized producer requests and resets whenever the backend restarts."
            steps={[
              "Configure the trusted producer with the server-side token.",
              "Post genuine observations to /api/v1/monitoring/observations.",
              "Review accepted and rejected request summaries here.",
            ]}
            actionLabel="View Model 2 readiness"
            onAction={() => onNavigate("model2-risk-page")}
          />
        )}
      </DashboardPanel>
      <DashboardPanel
        title="Recent production observations"
        subtitle="Model 1 predictions paired only with matching actual outcomes"
      >
        {observations.length ? (
          <div className="alert-history-table-wrap">
            <table className="alert-history-table">
              <thead>
                <tr>
                  <th>Observation ID</th>
                  <th>Batch</th>
                  <th>Predicted outcome</th>
                  <th>Positive probability</th>
                  <th>Actual outcome</th>
                  <th>Recorded</th>
                </tr>
              </thead>
              <tbody>
                {observations.map((observation) => (
                  <tr key={observation.observation_id}>
                    <td>{observation.observation_id}</td>
                    <td>{observation.batch_id}</td>
                    <td>{observation.predicted_class === 1 ? "Canceled" : "Not canceled"}</td>
                    <td>{formatPercent(observation.probability_positive)}</td>
                    <td>
                      {observation.actual_outcome == null
                        ? "Awaiting genuine outcome"
                        : observation.actual_outcome === 1
                          ? "Canceled"
                          : "Not canceled"}
                    </td>
                    <td>{formatDateTime(observation.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <DataWorkflowGuide
            title="No genuine production observations yet"
            description="Synthetic, offline, and CSV-imported records are intentionally excluded."
            steps={[
              "Send a genuine observation with the trusted producer token.",
              "Store its observation ID and returned prediction details.",
              "Post its verified outcome to make it eligible for evaluation.",
            ]}
            actionLabel="View Model 2 readiness"
            onAction={() => onNavigate("model2-risk-page")}
          />
        )}
      </DashboardPanel>
    </>
  );
}

function BatchExplorerPage({
  dashboard,
  days,
  onRefresh,
  refreshKey,
}: {
  dashboard: DashboardData;
  days: number;
  onRefresh: () => void;
  refreshKey: number;
}) {
  return (
    <>
      <section className="metric-card-row">
        <MetricSummary label="Real monitoring reports" value={formatNumber(dashboard.monitoring.report_count)} note={`Last ${days} days`} icon={<FileText size={22} />} />
        <MetricSummary
          label={dashboard.monitoring.report_count > 0 ? "Real batches" : "Dataset-simulation batches"}
          value={formatNumber(dashboard.monitoring.batch_count)}
          note="Source-qualified dashboard history"
          icon={<Boxes size={22} />}
        />
        <MetricSummary
          label="Latest batch"
          value={dashboard.latest_monitoring?.status ?? "No data"}
          note={dashboard.latest_monitoring?.batch_id ?? "No monitoring status"}
          icon={<Activity size={22} />}
        />
      </section>
      <DashboardPanel title="Run monitoring on a CSV batch" subtitle="Uses the existing authenticated monitoring workflow">
        <MonitoringCsvUpload onRefresh={onRefresh} />
      </DashboardPanel>
      <ReportsPage days={days} onRefresh={onRefresh} refreshKey={refreshKey} />
    </>
  );
}

function ModelPage({
  dashboard,
  onRefresh,
}: {
  dashboard: DashboardData;
  onRefresh: () => void;
}) {
  const metrics = [
    { label: "Accuracy", value: dashboard.metrics.accuracy, icon: <Target size={23} />, color: "cyan" },
    { label: "Precision", value: dashboard.metrics.precision, icon: <LayersIcon />, color: "purple" },
    { label: "Recall", value: dashboard.metrics.recall, icon: <Waves size={23} />, color: "purple" },
    { label: "F1-Score", value: dashboard.metrics.f1_score, icon: <BarChart3 size={23} />, color: "cyan" },
  ];

  return (
    <>
      <section className="metric-card-row" aria-label="Model evaluation metrics">
        {metrics.map((metric) => (
          <article className={`metric-card ${metric.color}`} key={metric.label}>
            <span className="metric-card-label">{metric.icon}{metric.label}</span>
            <strong>{metric.value.toFixed(3)}</strong>
            <small>{dashboard.model.validation_scope} · {formatNumber(dashboard.metrics.sample_count)} records</small>
          </article>
        ))}
        <article className="metric-card muted-metric">
          <span className="metric-card-label"><Activity size={23} />ROC-AUC</span>
          <strong>{dashboard.metrics.roc_auc.toFixed(3)}</strong>
          <small>{dashboard.model.validation_scope}</small>
        </article>
      </section>

      <section className="model-page-grid">
        <DashboardPanel
          title="Original Data Performance by Month"
          subtitle={`Held-out test rows · ${dashboard.model.dataset_period}`}
        >
          <HistoricalPerformanceChart history={dashboard.metrics.performance_history} />
        </DashboardPanel>
        <DashboardPanel title="Confusion Matrix" subtitle={`${formatNumber(dashboard.metrics.sample_count)} held-out records`}>
          <ConfusionMatrix matrix={dashboard.metrics.confusion_matrix} />
        </DashboardPanel>
        <DashboardPanel title="ROC Curve" subtitle="Receiver operating characteristic">
          <div className="roc-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={dashboard.metrics.roc_curve} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                <XAxis
                  type="number"
                  dataKey="false_positive_rate"
                  domain={[0, 1]}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                  tickFormatter={(value) => Number(value).toFixed(2)}
                  tick={{ fill: "#91a9c5", fontSize: 9 }}
                  axisLine={{ stroke: "#25395b" }}
                  tickLine={false}
                  label={{ value: "False positive rate", position: "insideBottom", offset: -2, fill: "#91a9c5", fontSize: 9 }}
                />
                <YAxis
                  type="number"
                  dataKey="true_positive_rate"
                  domain={[0, 1]}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                  tickFormatter={(value) => Number(value).toFixed(2)}
                  tick={{ fill: "#91a9c5", fontSize: 9 }}
                  axisLine={false}
                  tickLine={false}
                  width={38}
                />
                <Tooltip
                  formatter={(value, name) => [
                    Number(value).toFixed(3),
                    name === "false_positive_rate" ? "Random baseline" : "True positive rate",
                  ]}
                  contentStyle={{ background: "#0b1730", border: "1px solid #224474", borderRadius: 9, color: "#eff7ff" }}
                />
                <Line
                  dataKey="false_positive_rate"
                  name="Random baseline"
                  stroke="#71819b"
                  strokeDasharray="4 4"
                  strokeWidth={1}
                  dot={false}
                  isAnimationActive={false}
                />
                <Line
                  dataKey="true_positive_rate"
                  name="ROC curve"
                  stroke="#19c7ef"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="data-note model-evaluation-note">
            Held-out ROC-AUC: {dashboard.metrics.roc_auc.toFixed(3)}.
          </p>
        </DashboardPanel>
        <DashboardPanel title="Model Information" subtitle="Hotel booking cancellation model">
          <ModelInformation dashboard={dashboard} />
        </DashboardPanel>
        <DashboardPanel title="Feature Importance (Top 5)" subtitle={dashboard.model.algorithm}>
          <FeatureImportance features={dashboard.model.feature_importance} />
        </DashboardPanel>
        <DashboardPanel title="Prediction Summary" subtitle="Saved predictions">
          <PredictionSummary dashboard={dashboard} />
        </DashboardPanel>
      </section>
      <DashboardPanel title="Create Model 1 Predictions" subtitle="Upload a feature-only CSV; do not include actual labels">
        <PredictionCsvUpload onRefresh={onRefresh} />
      </DashboardPanel>
      <DashboardPanel
        title="Recent Model 1 prediction history"
        subtitle="Saved predictions for monitored batches · source classification is preserved"
      >
        {dashboard.predictions.prediction_history.length ? (
          <div className="alert-history-table-wrap">
            <table className="alert-history-table">
              <thead>
                <tr>
                  <th>Batch</th>
                  <th>Row / observation</th>
                  <th>Prediction</th>
                  <th>Cancellation probability</th>
                  <th>Model version</th>
                  <th>Source</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody>
                {dashboard.predictions.prediction_history.map((prediction) => (
                  <tr key={`${prediction.batch_id}:${prediction.observation_id ?? prediction.row_id}`}>
                    <td>{prediction.batch_id}</td>
                    <td>{prediction.observation_id ?? prediction.row_id}</td>
                    <td>{prediction.predicted_class === 1 ? "Canceled" : "Not canceled"}</td>
                    <td>{formatPercent(prediction.probability_positive)}</td>
                    <td>{prediction.model_version ?? "—"}</td>
                    <td>{prediction.source.replaceAll("_", " ")}</td>
                    <td>{formatDateTime(prediction.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState text="No saved predictions belong to a report in the selected period." />
        )}
      </DashboardPanel>
    </>
  );
}

function PredictionCsvUpload({ onRefresh }: { onRefresh: () => void }) {
  const [batchId, setBatchId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    setError("");
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(batchId.trim())) {
      setError("Use a batch ID with letters, numbers, dots, underscores or hyphens (maximum 128 characters).");
      return;
    }
    if (!file || file.size > 10 * 1024 * 1024) {
      setError("Choose a CSV file no larger than 10 MB.");
      return;
    }
    setSubmitting(true);
    try {
      const response = await apiFetch(
        `${API}/api/v1/predictions/cancellation/csv?batch_id=${encodeURIComponent(batchId.trim())}`,
        { method: "POST", headers: { "Content-Type": "text/csv" }, body: file },
      );
      const result = await response.json() as {
        batch_id?: string;
        prediction_count?: number;
        detail?: string | { message?: string };
      };
      if (!response.ok) {
        const detail = typeof result.detail === "string"
          ? result.detail
          : result.detail?.message ?? `Prediction request failed (HTTP ${response.status}).`;
        throw new Error(detail);
      }
      setMessage(`Stored ${formatNumber(result.prediction_count ?? 0)} predictions for ${result.batch_id}.`);
      setBatchId("");
      setFile(null);
      onRefresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not create predictions.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="workflow-upload-form" onSubmit={(event) => void submit(event)}>
      <label>Batch ID<input required maxLength={128} value={batchId} onChange={(event) => setBatchId(event.target.value)} placeholder="booking-batch-2026-11" /></label>
      <label>Feature CSV<input required type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label>
      <button type="submit" className="outline-action" disabled={submitting}>{submitting ? "Predicting…" : "Create predictions"}</button>
      {message && <p className="workflow-success" role="status">{message}</p>}
      {error && <p className="alert-error" role="alert">{error}</p>}
      <small>CSV columns must match the Model 1 feature schema. Actual labels are not accepted here.</small>
    </form>
  );
}

function LayersIcon() {
  return <Database size={23} />;
}

function FeatureImportance({
  features,
}: {
  features: { name: string; importance: number }[];
}) {
  const maximum = Math.max(...features.map((feature) => feature.importance), 0);

  return (
    <div className="importance-list">
      {features.map((feature) => (
        <div className="importance-row" key={feature.name}>
          <span title={displayFeatureName(feature.name)}>
            {displayFeatureName(feature.name)}
          </span>
          <i>
            <b
              style={{
                width: `${maximum > 0 ? (feature.importance / maximum) * 100 : 0}%`,
              }}
            />
          </i>
          <strong>{feature.importance.toFixed(3)}</strong>
        </div>
      ))}
      <p className="data-note">
        Training-set impurity importance; descriptive, not causal.
      </p>
    </div>
  );
}

function HistoricalPerformanceChart({
  history,
}: {
  history: DashboardData["metrics"]["performance_history"] | undefined;
}) {
  if (!Array.isArray(history) || history.length === 0) {
    return <EmptyState text="DEMO · No dated held-out booking rows are available." />;
  }

  return (
    <>
      <div className="history-chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={history} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="#25395b" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="period"
              tick={{ fill: "#91a9c5", fontSize: 9 }}
              axisLine={{ stroke: "#25395b" }}
              tickLine={false}
              minTickGap={18}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tickFormatter={(value) => Number(value).toFixed(2)}
              tick={{ fill: "#91a9c5", fontSize: 9 }}
              axisLine={false}
              tickLine={false}
              width={38}
            />
            <Tooltip
              formatter={(value, name) => [
                Number(value).toFixed(3),
                String(name).replaceAll("_", " "),
              ]}
              labelFormatter={(label, payload) =>
                `${String(label)} · ${formatNumber(payload[0]?.payload.sample_count)} held-out rows`
              }
              contentStyle={{ background: "#0b1730", border: "1px solid #224474", borderRadius: 9, color: "#eff7ff" }}
            />
            <Line dataKey="accuracy" name="Accuracy" stroke="#168cff" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="precision" name="Precision" stroke="#d34bff" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="recall" name="Recall" stroke="#0ae2c1" strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="f1_score" name="F1 score" stroke="#ffae25" strokeWidth={2} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="data-note model-evaluation-note">
        Retrospective slices of the original dataset&apos;s random held-out test
        rows; not live monitoring or a time-based model validation.
      </p>
    </>
  );
}

function DataDriftPage({
  dashboard,
  days,
  refreshKey,
}: {
  dashboard: DashboardData;
  days: number;
  refreshKey: number;
}) {
  const [source, setSource] = useState<"real" | "unknown" | "synthetic_test">("synthetic_test");
  const reportSource = source === "synthetic_test" ? "SIMULATED" : source.toUpperCase();
  const { reports, error, loading } = useMonitoringReports(days, reportSource, refreshKey);
  const historicalReports = [...reports].reverse();
  const insight = dashboard.drift.insights_by_source?.[source] ?? null;
  const features = insight?.drifted_features ?? [];
  const warningThreshold = insight?.thresholds.psi_warning ?? dashboard.drift.warning_threshold ?? 0.1;
  const criticalThreshold = insight?.thresholds.psi_critical ?? dashboard.drift.threshold ?? 0.2;
  const maxScore = Math.max(
    criticalThreshold,
    ...features.map((feature) => feature.psi ?? 0),
    0.01,
  );

  return (
    <>
      <section className="metric-card-row drift-kpis">
        <MetricSummary label="Drifted Features" value={String(features.filter((feature) => feature.severity !== "HEALTHY").length)} note={`Warning threshold ${warningThreshold.toFixed(2)}`} icon={<AlertTriangle size={22} />} />
        <MetricSummary label="Critical Features" value={String(features.filter((feature) => feature.severity === "CRITICAL").length)} note="PSI critical or critical quality evidence" icon={<Activity size={22} />} />
        <MetricSummary label="Batch Status" value={insight?.status ?? "No data"} note={insight?.batch_id ?? "No report for this source"} icon={<Database size={22} />} />
        <MetricSummary label="Data Source" value={source.replaceAll("_", " ")} note="Provenance remains visible; simulated data is separate" icon={<Waves size={22} />} />
      </section>
      <section className="drift-source-control">
        <label htmlFor="drift-source">Monitoring source</label>
        <select id="drift-source" value={source} onChange={(event) => setSource(event.target.value as typeof source)}>
          <option value="real">Real declaration</option>
          <option value="unknown">Unknown provenance</option>
          <option value="synthetic_test">Simulated (separate)</option>
        </select>
        {insight && <span>{insight.batch_id} · {formatDateTime(insight.timestamp)}</span>}
      </section>
      <DashboardPanel
        title="Historical average PSI"
        subtitle={`Saved ${source.replaceAll("_", " ")} monitoring reports · selected period`}
      >
        {error && <p className="alert-error" role="alert">{error}</p>}
        {loading ? (
          <EmptyState text="Loading persisted drift history…" />
        ) : historicalReports.some((report) => report.drift_summary.average_psi != null) ? (
          <div className="history-trend-chart">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={historicalReports.filter((report) => report.drift_summary.average_psi != null)}>
                <CartesianGrid stroke="#27334b" strokeDasharray="3 5" vertical={false} />
                <XAxis
                  dataKey="timestamp"
                  tickFormatter={(value) => formatDate(String(value))}
                  tick={{ fill: "#8290aa", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, "auto"]}
                  tick={{ fill: "#8290aa", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  width={42}
                />
                <Tooltip
                  labelFormatter={(value) => formatDateTime(String(value))}
                  formatter={(value) => [Number(value).toFixed(3), "Average PSI"]}
                  contentStyle={{ background: "#111827", border: "1px solid #29364e", borderRadius: 10, color: "#e9edf7" }}
                />
                <Line
                  dataKey="drift_summary.average_psi"
                  name="Average PSI"
                  stroke="#20d9f3"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyState text={`No usable PSI trend is available for ${source.replaceAll("_", " ")} reports in this period.`} />
        )}
      </DashboardPanel>
      <section className="two-column-page-grid">
        <DashboardPanel title="Feature Drift Ranking" subtitle="Severity-ranked PSI and KS evidence for the selected source">
          {features.length ? (
            <div className="drift-feature-list detailed-drift-list">
              {features.map((feature) => (
                <article className="drift-feature-detail" key={feature.feature}>
                  <div className="drift-feature-heading">
                    <strong>{displayFeatureName(feature.feature)}</strong>
                    <span className={`severity-pill ${feature.severity.toLowerCase()}`}>{feature.severity}</span>
                  </div>
                  <p>{feature.change_summary}</p>
                  <div className="drift-stat-row">
                    <span>PSI <strong>{feature.psi?.toFixed(3) ?? "—"}</strong></span>
                    <span>KS <strong>{feature.ks_statistic?.toFixed(3) ?? "—"}</strong></span>
                    <span>p-value <strong>{feature.ks_p_value?.toExponential(2) ?? "—"}</strong></span>
                    <span>{feature.feature_type}</span>
                  </div>
                  <span className="drift-track"><i className={(feature.psi ?? 0) >= warningThreshold ? "threshold-crossed" : ""} style={{ width: `${Math.min(100, ((feature.psi ?? 0) / maxScore) * 100)}%` }} /></span>
                  {feature.root_causes.length > 0 && (
                    <ul className="drift-cause-list">
                      {feature.root_causes.map((cause, index) => (
                        <li key={`${cause.type}-${index}`}>
                          <strong>{cause.type.replaceAll("_", " ")}</strong>
                          <span>{cause.description}</span>
                          <small>{cause.confidence === "suggested" ? "Possible cause" : "Observed evidence"} · {JSON.stringify(cause.evidence)}</small>
                        </li>
                      ))}
                    </ul>
                  )}
                  {feature.reference_distribution && feature.current_distribution && (
                    <details className="distribution-details">
                      <summary>Reference and current distributions</summary>
                      <div className="distribution-pair">
                        <div><strong>Reference</strong><pre>{JSON.stringify(feature.reference_distribution, null, 2)}</pre></div>
                        <div><strong>Current</strong><pre>{JSON.stringify(feature.current_distribution, null, 2)}</pre></div>
                      </div>
                    </details>
                  )}
                </article>
              ))}
            </div>
          ) : (
            <div className="chart-placeholder">
              <Waves size={28} />
              <strong>No drift report for this source</strong>
              <span>Feature explanations appear after a monitoring run is saved.</span>
            </div>
          )}
        </DashboardPanel>
        <DashboardPanel title="Latest Batch" subtitle="Source and monitoring status">
          {insight ? (
            <div className="detail-list">
              <DetailRow label="Batch ID" value={insight.batch_id} />
              <DetailRow label="Monitored" value={formatDateTime(insight.timestamp)} />
              <DetailRow label="Status" value={insight.status} state={insight.status.toLowerCase()} />
              <DetailRow label="Source" value={insight.data_source} />
            </div>
          ) : <EmptyState text={`No ${source.replaceAll("_", " ")} monitoring report is available for this period.`} />}
          {insight && (
            <div className="recommendation-list">
              <strong>Recommended actions</strong>
              {insight.recommendations.map((recommendation) => <p key={recommendation}>{recommendation}</p>)}
            </div>
          )}
        </DashboardPanel>
      </section>
      <section className="quality-kpi-row">
        <QualityBar label="Missing values" value={insight?.quality_summary.missing_percentage ?? null} />
        <QualityBar label="Duplicate records" value={insight?.quality_summary.duplicate_percentage ?? null} />
        <QualityBar label="Invalid values" value={insight?.quality_summary.invalid_percentage ?? null} />
      </section>
    </>
  );
}

function PerformancePage({
  dashboard,
  onRefresh,
  onNavigate,
}: {
  dashboard: DashboardData;
  onRefresh: () => void;
  onNavigate: (target: string) => void;
}) {
  const [batchId, setBatchId] = useState("");
  const [outcomeFile, setOutcomeFile] = useState<File | null>(null);
  const [uploadMessage, setUploadMessage] = useState("");
  const [uploadError, setUploadError] = useState("");
  const [uploading, setUploading] = useState(false);
  const outcomeFileInputRef = useRef<HTMLInputElement>(null);
  const distribution = [
    { name: "Not canceled", value: dashboard.predictions.class_counts.not_cancelled },
    { name: "Canceled", value: dashboard.predictions.class_counts.cancelled },
  ].filter((item) => item.value > 0);
  const outcomeHistory = dashboard.operational_performance.history;
  const latestOutcome = outcomeHistory.at(-1);

  async function importOutcomes(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUploadMessage("");
    setUploadError("");
    if (!batchId.trim() || !outcomeFile) {
      setUploadError("Enter a batch ID and choose an outcomes CSV file.");
      return;
    }

    setUploading(true);
    try {
      const response = await apiFetch(
        `${API}/api/v1/outcomes/${encodeURIComponent(batchId.trim())}/csv`,
        {
          method: "POST",
          headers: { "Content-Type": "text/csv" },
          body: await outcomeFile.text(),
        },
      );
      const result = (await response.json()) as {
        detail?: string;
        outcome_count?: number;
      };
      if (!response.ok) {
        throw new Error(
          typeof result.detail === "string"
            ? result.detail
            : `Outcome import failed (HTTP ${response.status}).`,
        );
      }

      setUploadMessage(
        `Imported ${formatNumber(result.outcome_count)} outcomes for ${batchId.trim()}.`,
      );
      setOutcomeFile(null);
      if (outcomeFileInputRef.current) outcomeFileInputRef.current.value = "";
      onRefresh();
    } catch (error) {
      setUploadError(
        error instanceof Error ? error.message : "Could not import outcome CSV.",
      );
    } finally {
      setUploading(false);
    }
  }

  return (
    <>
      <section className="metric-card-row">
        <MetricSummary label="Accuracy" value={dashboard.metrics.accuracy.toFixed(3)} note="Held-out test set" icon={<Target size={22} />} />
        <MetricSummary label="Precision" value={dashboard.metrics.precision.toFixed(3)} note="Positive class" icon={<Database size={22} />} />
        <MetricSummary label="Recall" value={dashboard.metrics.recall.toFixed(3)} note="Positive class" icon={<Waves size={22} />} />
        <MetricSummary label="F1-Score" value={dashboard.metrics.f1_score.toFixed(3)} note="Positive class" icon={<BarChart3 size={22} />} />
        <MetricSummary label="Evaluated Records" value={formatNumber(dashboard.metrics.sample_count)} note={dashboard.model.validation_scope} icon={<FileText size={22} />} />
      </section>
      <section className="two-column-page-grid performance-detail-grid">
        <DashboardPanel
          title="Original Data Held-out Evaluation"
          subtitle={`Model 1 · ${formatNumber(dashboard.metrics.sample_count)} held-out test records`}
        >
          <div className="performance-bars">
            {[
              ["Accuracy", dashboard.metrics.accuracy],
              ["Precision", dashboard.metrics.precision],
              ["Recall", dashboard.metrics.recall],
              ["F1-Score", dashboard.metrics.f1_score],
            ].map(([label, value]) => (
              <div className="performance-bar" key={String(label)}>
                <span>{label}</span>
                <i><b style={{ width: `${Math.max(0, Math.min(100, Number(value) * 100))}%` }} /></i>
                <strong>{Number(value).toFixed(3)}</strong>
              </div>
            ))}
          </div>
          <p className="data-note">These are the model&apos;s recorded held-out evaluation results, not a live historical trend.</p>
        </DashboardPanel>
        <DashboardPanel title="Confusion Matrix" subtitle="Actual vs. predicted class">
          <ConfusionMatrix matrix={dashboard.metrics.confusion_matrix} />
        </DashboardPanel>
        <DashboardPanel title="Prediction Distribution" subtitle="Saved prediction records">
          {distribution.length && dashboard.predictions.distribution_sample_count ? (
            <div className="performance-distribution">
              <div className="performance-donut">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius="68%" outerRadius="92%" paddingAngle={3} stroke="none">
                      {distribution.map((item, index) => <Cell key={item.name} fill={index === 0 ? "#19c9eb" : "#a568ff"} />)}
                    </Pie>
                    <Tooltip formatter={(value) => [formatNumber(Number(value)), "Predictions"]} />
                  </PieChart>
                </ResponsiveContainer>
                <strong>{formatNumber(dashboard.predictions.distribution_sample_count)}</strong>
              </div>
              <div className="distribution-legend">
                {distribution.map((item, index) => (
                  <div className="distribution-item" key={item.name}>
                    <span className={`legend-dot ${index === 0 ? "blue" : "purple"}`} />
                    <span>{item.name}</span>
                    <strong>{formatNumber(item.value)}</strong>
                  </div>
                ))}
              </div>
            </div>
          ) :           <EmptyState text="DEMO · No saved predictions are available for this period." />}
        </DashboardPanel>
        <DashboardPanel
          title="Original Data Performance by Month"
          subtitle={`Held-out test rows · ${dashboard.model.dataset_period}`}
        >
          <HistoricalPerformanceChart history={dashboard.metrics.performance_history} />
        </DashboardPanel>
      </section>
      <section className="two-column-page-grid performance-detail-grid">
        <DashboardPanel
          title="PRODUCTION Batch Performance"
          subtitle={`${formatNumber(dashboard.operational_performance.evaluated_record_count)} token-authenticated outcomes across ${formatNumber(dashboard.operational_performance.evaluated_batch_count)} production batches`}
        >
          {outcomeHistory.length ? (
            <>
              <div className="history-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={outcomeHistory} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
                    <CartesianGrid stroke="#25395b" strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="batch_id" tick={{ fill: "#91a9c5", fontSize: 9 }} axisLine={{ stroke: "#25395b" }} tickLine={false} minTickGap={18} />
                    <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={(value) => Number(value).toFixed(2)} tick={{ fill: "#91a9c5", fontSize: 9 }} axisLine={false} tickLine={false} width={38} />
                    <Tooltip
                      formatter={(value, name) => [Number(value).toFixed(3), String(name).replaceAll("_", " ")]}
                      labelFormatter={(label, payload) => `${String(label)} · ${formatDateTime(payload[0]?.payload.timestamp)}`}
                      contentStyle={{ background: "#0b1730", border: "1px solid #224474", borderRadius: 9, color: "#eff7ff" }}
                    />
                    <Line dataKey="accuracy" name="Accuracy" stroke="#168cff" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                    <Line dataKey="precision" name="Precision" stroke="#d34bff" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                    <Line dataKey="recall" name="Recall" stroke="#0ae2c1" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                    <Line dataKey="f1_score" name="F1 score" stroke="#ffae25" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="data-note">Calculated only from token-authenticated outcomes matched to stored PRODUCTION predictions.</p>
            </>
          ) : (
            <DataWorkflowGuide
              title="No verified production evaluations yet"
              description="Production metrics appear only when real outcomes are matched to token-authenticated predictions."
              steps={[
                "Send a real batch through the trusted producer API.",
                "Submit verified outcomes for its prediction IDs.",
                "Review accuracy, precision, recall, and F1 here.",
              ]}
              actionLabel="Open Production Data"
              onAction={() => onNavigate("ingestion-page")}
            />
          )}
        </DashboardPanel>
        <DashboardPanel
          title="Import Non-Production Actual Booking Outcomes"
          subtitle="CSV columns: row_id,is_canceled"
        >
          <form className="outcome-import-form" onSubmit={(event) => void importOutcomes(event)}>
            <label>
              <span>Monitored batch ID</span>
              <input
                type="text"
                required
                value={batchId}
                onChange={(event) => setBatchId(event.target.value)}
                placeholder="Exact batch ID from Model 1 predictions"
              />
            </label>
            <label>
              <span>Outcomes CSV</span>
              <input
                type="file"
                required
                accept=".csv,text/csv"
                ref={outcomeFileInputRef}
                onChange={(event) => setOutcomeFile(event.target.files?.[0] ?? null)}
              />
            </label>
            <p className="data-note">
              CSV imports are for offline or operator-declared real batches only. Production outcomes must use the token-authenticated production API; CSV/session outcomes cannot qualify Model 2 training labels.
            </p>
            <button type="submit" className="outline-action" disabled={uploading}>
              {uploading ? "Importing outcomes…" : "Import and evaluate"}
            </button>
            {uploadError && <p className="outcome-import-message error" role="alert">{uploadError}</p>}
            {uploadMessage && <p className="outcome-import-message success" role="status">{uploadMessage}</p>}
          </form>
        </DashboardPanel>
        <DashboardPanel title="Latest PRODUCTION Evaluation" subtitle="Token-authenticated actual outcomes only">
          {latestOutcome ? (
            <>
              <div className="detail-list">
                <DetailRow label="Batch ID" value={latestOutcome.batch_id} />
                <DetailRow label="Evaluation time" value={formatDateTime(latestOutcome.timestamp)} />
                <DetailRow label="Labeled records" value={formatNumber(latestOutcome.sample_count)} />
                <DetailRow label="Model version" value={latestOutcome.model_version} />
              </div>
              <ConfusionMatrix matrix={latestOutcome.confusion_matrix} />
            </>
          ) : (
            <>
              <div className="detail-list">
                <DetailRow label="Batch ID" value="Waiting for first production batch" />
                <DetailRow label="Evaluation time" value="Not available yet" />
                <DetailRow label="Verified labeled records" value={formatNumber(dashboard.operational_performance.evaluated_record_count)} />
                <DetailRow label="Model version" value="Available after evaluation" />
              </div>
              <DataWorkflowGuide
                title="A confusion matrix needs verified labels"
                description="CSV imports remain non-production and cannot populate this panel."
                steps={[
                  "Record real producer observations.",
                  "Submit the matching genuine outcomes.",
                  "The latest evaluation and confusion matrix will appear here.",
                ]}
                actionLabel="Open Production Data"
                onAction={() => onNavigate("ingestion-page")}
              />
            </>
          )}
        </DashboardPanel>
        <DashboardPanel title="Outcome Collection" subtitle="Explicitly real batches in the selected period">
          <div className="coverage-summary">
            <strong>{formatNumber(dashboard.operational_performance.pending_outcome_count)}</strong>
            <span>predictions still awaiting actual outcomes</span>
            <strong>{formatNumber(dashboard.operational_performance.evaluated_record_count)}</strong>
            <span>predictions evaluated against received genuine labels</span>
          </div>
        </DashboardPanel>
      </section>
    </>
  );
}

function ReportsPage({
  days,
  onRefresh,
  refreshKey,
}: {
  days: number;
  onRefresh: () => void;
  refreshKey: number;
}) {
  const [source, setSource] = useState("");
  const { reports, error, loading } = useMonitoringReports(days, source, refreshKey);

  return (
    <>
      <section className="metric-card-row">
        <MetricSummary label="Monitoring Reports" value={formatNumber(reports.length)} note={`Last ${days} days · selected source`} icon={<FileText size={22} />} />
        <MetricSummary label="Monitored Batches" value={formatNumber(new Set(reports.map((report) => report.batch_id)).size)} note="Distinct batches" icon={<Database size={22} />} />
        <MetricSummary label="Critical Reports" value={formatNumber(reports.filter((report) => report.status === "CRITICAL").length)} note="Monitoring status only" icon={<Bell size={22} />} />
      </section>
      <DashboardPanel title="Monitoring Report History" subtitle="Saved reports can be downloaded as JSON">
        <div className="alert-filters">
          <label>Provenance<select value={source} onChange={(event) => setSource(event.target.value)}><option value="">All sources</option><option value="REAL">Real</option><option value="SIMULATED">Simulated</option><option value="UNKNOWN">Unknown</option></select></label>
        </div>
        {error && <p className="alert-error" role="alert">{error}</p>}
        {loading ? <EmptyState text="Loading monitoring reports…" /> : reports.length ? (
          <div className="alert-history-table-wrap">
            <table className="alert-history-table">
              <thead><tr><th>Status</th><th>Quality</th><th>Batch</th><th>Rows</th><th>Source</th><th>Timestamp</th><th>Report</th></tr></thead>
              <tbody>
                {reports.map((report) => (
                  <tr key={`${report.batch_id}:${report.timestamp}`}>
                    <td><span className={`severity-pill ${report.status.toLowerCase()}`}>{report.status}</span></td>
                    <td>{report.quality_status}</td><td>{report.batch_id}</td><td>{formatNumber(report.rows)}</td>
                    <td>{report.source}</td><td>{formatDateTime(report.timestamp)}</td>
                    <td><a className="text-button" href={`${API}${report.download_url}`} download>Download JSON</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <EmptyState text="No monitoring reports match the selected source and period." />}
      </DashboardPanel>
      <div className="report-refresh-row">
        <span>Reports contain monitoring evidence only; performance metrics require verified actual labels.</span>
        <button type="button" className="outline-action" onClick={onRefresh}><Activity size={15} /> Refresh reports</button>
      </div>
    </>
  );
}

function AlertsPage({
  dashboard,
  refreshKey,
}: {
  dashboard: DashboardData;
  refreshKey: number;
}) {
  const [alerts, setAlerts] = useState<MonitoringAlert[]>([]);
  const [severity, setSeverity] = useState("");
  const [source, setSource] = useState("");
  const [batchId, setBatchId] = useState("");
  const [periodDays, setPeriodDays] = useState(dashboard.monitoring.period_days);
  const [selectedAlertId, setSelectedAlertId] = useState("");
  const [loadedQuery, setLoadedQuery] = useState("");
  const [error, setError] = useState("");
  const queryKey = [periodDays, severity, source, batchId.trim(), refreshKey].join("|");

  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({ period_days: String(periodDays), limit: "1000" });
    if (severity) params.set("severity", severity);
    if (source) params.set("source", source);
    if (batchId.trim()) params.set("batch_id", batchId.trim());
    apiFetch(`${API}/api/v1/alerts?${params}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Alert history request failed (${response.status}).`);
        return response.json() as Promise<{ alerts: MonitoringAlert[] }>;
      })
      .then((result) => {
        setError("");
        setAlerts(result.alerts);
        setSelectedAlertId((current) =>
          result.alerts.some((alert) => alert.alert_id === current)
            ? current
            : result.alerts[0]?.alert_id ?? "",
        );
        setLoadedQuery(queryKey);
      })
      .catch((fetchError: unknown) => {
        if (fetchError instanceof DOMException && fetchError.name === "AbortError") return;
        setError(fetchError instanceof Error ? fetchError.message : "Unable to load alert history.");
        setAlerts([]);
        setLoadedQuery(queryKey);
      });
    return () => controller.abort();
  }, [batchId, periodDays, queryKey, severity, source]);

  const selectedAlert = alerts.find((alert) => alert.alert_id === selectedAlertId);
  const loading = loadedQuery !== queryKey;
  const countBySeverity = (level: MonitoringAlert["severity"]) =>
    alerts.filter((alert) => alert.severity === level).length;

  return (
    <>
      <section className="metric-card-row">
        <MetricSummary label="Total Alerts" value={formatNumber(alerts.length)} note="Matching history filters" icon={<Bell size={22} />} />
        <MetricSummary label="Critical" value={formatNumber(countBySeverity("CRITICAL"))} note="Immediate investigation" icon={<AlertTriangle size={22} />} />
        <MetricSummary label="Warnings" value={formatNumber(countBySeverity("WARNING"))} note="Review recommended" icon={<Activity size={22} />} />
        <MetricSummary label="Healthy" value={formatNumber(countBySeverity("HEALTHY"))} note="Successful monitored batches" icon={<Check size={22} />} />
      </section>
      <DashboardPanel title="Historical Alert Filters" subtitle="Persisted monitoring alerts; source provenance is never merged">
        <div className="alert-filters">
          <label>Severity<select value={severity} onChange={(event) => setSeverity(event.target.value)}><option value="">All severities</option><option>CRITICAL</option><option>WARNING</option><option>HEALTHY</option></select></label>
          <label>Source<select value={source} onChange={(event) => setSource(event.target.value)}><option value="">All sources</option><option>REAL</option><option>SIMULATED</option><option>UNKNOWN</option></select></label>
          <label>Period<select value={periodDays} onChange={(event) => setPeriodDays(Number(event.target.value))}>{PERIODS.map((period) => <option key={period.days} value={period.days}>{period.label}</option>)}</select></label>
          <label>Batch ID<input value={batchId} onChange={(event) => setBatchId(event.target.value)} placeholder="Filter by exact batch ID" /></label>
        </div>
        {error && <p className="alert-error" role="alert">{error}</p>}
        {loading ? <EmptyState text="Loading persisted alert history…" /> : alerts.length ? (
          <div className="alert-history-layout">
            <div className="alert-history-table-wrap">
              <table className="alert-history-table">
                <thead><tr><th>Severity</th><th>Feature</th><th>Batch</th><th>Source</th><th>Detected</th></tr></thead>
                <tbody>
                  {alerts.map((alert) => (
                    <tr key={alert.alert_id} className={alert.alert_id === selectedAlertId ? "selected" : ""}>
                      <td><button className={`severity-pill ${alert.severity.toLowerCase()}`} onClick={() => setSelectedAlertId(alert.alert_id)}>{alert.severity}</button></td>
                      <td><button className="alert-table-select" onClick={() => setSelectedAlertId(alert.alert_id)}>{displayFeatureName(alert.affected_feature)}</button></td>
                      <td>{alert.batch_id}</td><td>{alert.data_source}</td><td>{formatDateTime(alert.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {selectedAlert && (
              <aside className="alert-detail-card">
                <div className="drift-feature-heading"><strong>Alert details</strong><span className={`severity-pill ${selectedAlert.severity.toLowerCase()}`}>{selectedAlert.severity}</span></div>
                <div className="detail-list">
                  <DetailRow label="Alert ID" value={selectedAlert.alert_id} />
                  <DetailRow label="Batch ID" value={selectedAlert.batch_id} />
                  <DetailRow label="Detected" value={formatDateTime(selectedAlert.timestamp)} />
                  <DetailRow label="Affected feature" value={selectedAlert.affected_feature} />
                  <DetailRow label="Data source" value={selectedAlert.data_source} />
                </div>
                <p><strong>Detection reason</strong><br />{selectedAlert.detection_reason}</p>
                <p><strong>Recommended action</strong><br />{selectedAlert.recommended_action}</p>
                <details><summary>Evidence</summary><pre>{JSON.stringify(selectedAlert.evidence, null, 2)}</pre></details>
              </aside>
            )}
          </div>
        ) : <EmptyState text="No alerts match the selected filters and period." />}
      </DashboardPanel>
    </>
  );
}

function SettingsPage({
  dashboard,
  autoRefresh,
  setAutoRefresh,
  days,
  connection,
}: {
  dashboard: DashboardData;
  autoRefresh: boolean;
  setAutoRefresh: (value: boolean) => void;
  days: number;
  connection: "connecting" | "online" | "offline";
}) {
  const connectionLabel = {
    connecting: "CONNECTING",
    online: "ONLINE",
    offline: "OFFLINE",
  }[connection];
  return (
    <section className="settings-page-grid">
      <DashboardPanel title="Dashboard Preferences" subtitle="Local display and refresh settings">
        <label className="settings-page-toggle">
          <span>
            <strong>Automatic refresh</strong>
            <small>Check for new model data every 30 seconds</small>
          </span>
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(event) => setAutoRefresh(event.target.checked)}
          />
        </label>
        <div className="detail-list">
          <DetailRow label="Selected period" value={PERIODS.find((period) => period.days === days)?.label ?? "Custom"} />
          <DetailRow label="API base URL" value={API || "Same-origin proxy"} />
          <DetailRow label="Refresh interval" value={autoRefresh ? "30 seconds" : "Paused"} />
        </div>
      </DashboardPanel>
      <DashboardPanel title="Current Environment" subtitle="Status reported by the backend">
        <div className="detail-list">
          <DetailRow label="Dashboard/API connection" value={connectionLabel} state={connection === "online" ? "good" : connection === "offline" ? "bad" : "pending"} />
          <DetailRow label="Application health" value={dashboard.system.application_status} state={dashboard.system.application_status === "OK" ? "good" : "bad"} />
          <DetailRow label="Model 1 readiness" value={dashboard.system.model1_readiness} state={dashboard.system.model1_readiness === "READY" ? "good" : "bad"} />
          <DetailRow label="Feature schema readiness" value={dashboard.system.schema_readiness} state={dashboard.system.schema_readiness === "READY" ? "good" : "bad"} />
          <DetailRow label="Model 2 PRODUCTION readiness" value={productionModel2StatusLabel(dashboard.model2)} state={dashboard.model2.status === "TRAINED" ? "good" : "pending"} />
          <DetailRow
            label="Production ingestion"
            value={dashboard.system.production_ingestion_configured ? "CONFIGURED" : "NOT CONFIGURED"}
            state={dashboard.system.production_ingestion_configured ? "good" : "pending"}
          />
          <DetailRow label="Model 1 artifact" value={dashboard.system.model_artifact} state={dashboard.system.model_artifact === "AVAILABLE" ? "good" : "bad"} />
          <DetailRow label="Monitoring reports" value={dashboard.system.monitoring_reports} />
          <DetailRow label="Live outcomes" value={dashboard.system.live_outcomes} />
        </div>
        <p className="data-note">
          Only secret-configuration presence is reported. Tokens, passwords and
          signing keys are never returned by the health or dashboard APIs.
        </p>
      </DashboardPanel>
    </section>
  );
}

function MonitoringPage({
  dashboard,
  onRefresh,
  refreshKey,
}: {
  dashboard: DashboardData;
  onRefresh: () => void;
  refreshKey: number;
}) {
  const developmentAvailable =
    dashboard.model2_development.status === "AVAILABLE";
  const systemRows = [
    { label: "API service", value: dashboard.system.api, state: dashboard.system.api === "ONLINE" ? "good" : "bad" },
    { label: "Model artifact", value: dashboard.system.model_artifact, state: dashboard.system.model_artifact === "AVAILABLE" ? "good" : "bad" },
    { label: "Monitoring reports", value: dashboard.system.monitoring_reports, state: dashboard.system.monitoring_reports === "AVAILABLE" ? "good" : "pending" },
    { label: "Live outcomes", value: dashboard.system.live_outcomes, state: dashboard.system.live_outcomes === "AVAILABLE" ? "good" : "pending" },
    { label: "Model 2 PRODUCTION", value: productionModel2StatusLabel(dashboard.model2), state: dashboard.model2.status === "TRAINED" ? "good" : "pending" },
  ];

  return (
    <>
      <section className="metric-card-row">
        <MetricSummary label="API Service" value={dashboard.system.api} note="Dashboard API status" icon={<Activity size={22} />} />
        <MetricSummary label="Model Artifact" value={dashboard.system.model_artifact} note={dashboard.model.status} icon={<BrainCircuit size={22} />} />
        <MetricSummary label="Genuine Observations" value={formatNumber(dashboard.production_observations.total_observations)} note="Trusted production-source Model 1 predictions" icon={<FileText size={22} />} />
        <MetricSummary label="Labeled Outcomes" value={formatNumber(dashboard.production_observations.labeled_observations)} note="Matched actual cancellation labels" icon={<Check size={22} />} />
        <MetricSummary label="Pending Outcomes" value={formatNumber(dashboard.production_observations.pending_outcomes)} note="Production predictions awaiting labels" icon={<Target size={22} />} />
        <MetricSummary label="Latest Batch" value={dashboard.latest_monitoring?.status ?? "No data"} note={dashboard.latest_monitoring?.batch_id ?? "No monitored batches"} icon={<Database size={22} />} />
      </section>
      <DashboardPanel title="Run Monitoring on a CSV Batch" subtitle="Creates a timestamped report and persists deduplicated alerts">
        <MonitoringCsvUpload onRefresh={onRefresh} />
      </DashboardPanel>
      <DashboardPanel
        title="PRODUCTION Model 2 Risk Model"
        subtitle="Genuine production observations only; development data is separate and never counts toward eligibility"
      >
        <div className={`report-status-banner ${dashboard.model2.status === "TRAINED" ? "healthy" : "warning"}`}>
          <BrainCircuit size={20} />
          <span>
            <strong>PRODUCTION · {productionModel2StatusLabel(dashboard.model2)}</strong>
            <small>
              {dashboard.model2.labeled_observation_count}/
              {dashboard.model2.minimum_labeled_observations} labeled production batches
            </small>
          </span>
        </div>
        <div className="detail-list">
          <DetailRow
            label="Production batches collected"
            value={formatNumber(dashboard.model2.production_batch_count)}
          />
          <DetailRow
            label={`Labeled batches (required: ${dashboard.model2.minimum_labeled_observations})`}
            value={`${formatNumber(dashboard.model2.labeled_batch_count)}/${dashboard.model2.minimum_labeled_observations}`}
          />
          <DetailRow
            label="Unlabeled production batches"
            value={formatNumber(dashboard.model2.unlabeled_batch_count)}
          />
          <DetailRow
            label="Training eligibility"
            value={dashboard.model2.training_eligibility === "NOT_TRAINED"
              ? productionModel2StatusLabel(dashboard.model2)
              : dashboard.model2.training_eligibility.replaceAll("_", " ")}
          />
          <DetailRow
            label="Collection progress"
            value={`${dashboard.model2.training_progress_percent.toFixed(0)}%`}
          />
          <DetailRow
            label="Last training result"
            value={dashboard.model2.latest_training_result
              ? dashboard.model2.latest_training_result.training_performed
                ? "Model trained"
                : dashboard.model2.latest_training_result.status === "NOT_TRAINED"
                  ? "Not trained"
                  : dashboard.model2.latest_training_result.status?.replaceAll("_", " ") ?? "No model trained"
              : "No training attempt"}
          />
          <DetailRow
            label="Last training attempt"
            value={dashboard.model2.latest_training_result?.timestamp
              ? formatDateTime(dashboard.model2.latest_training_result.timestamp)
              : "Not attempted"}
          />
        </div>
        <progress
          className="model2-training-progress"
          aria-label="Model 2 labeled production batch collection progress"
          max={100}
          value={dashboard.model2.training_progress_percent}
        />
        {dashboard.model2.latest_training_result?.reason && (
          <p className="data-note">
            Last training result: {dashboard.model2.latest_training_result.reason}
          </p>
        )}
        {dashboard.model2.status === "TRAINED" ? (
          <div className="detail-list">
            <DetailRow label="Model version" value={dashboard.model2.model_version ?? "—"} />
            <DetailRow label="Trained" value={formatDateTime(dashboard.model2.training_timestamp)} />
            <DetailRow
              label="Holdout accuracy"
              value={dashboard.model2.evaluation_metrics ? `${(dashboard.model2.evaluation_metrics.accuracy * 100).toFixed(1)}%` : "—"}
            />
            <DetailRow
              label="Holdout recall"
              value={dashboard.model2.evaluation_metrics ? `${(dashboard.model2.evaluation_metrics.recall * 100).toFixed(1)}%` : "—"}
            />
            <DetailRow
              label="Holdout F1"
              value={dashboard.model2.evaluation_metrics ? dashboard.model2.evaluation_metrics.f1_score.toFixed(3) : "—"}
            />
          </div>
        ) : (
          <>
            <div className="detail-list">
              <DetailRow label="Model version" value="Not available" />
              <DetailRow label="Trained" value="Not trained" />
              <DetailRow label="Holdout accuracy" value="Not evaluated" />
              <DetailRow label="Holdout recall" value="Not evaluated" />
              <DetailRow label="Holdout F1" value="Not evaluated" />
            </div>
            <p className="data-note">
              {dashboard.model2.reason ??
                dashboard.model2.latest_training_result?.reason ??
                "Collecting eligible production batch labels before training."}
            </p>
          </>
        )}
      </DashboardPanel>
      <Model2DevelopmentRiskAssessment
        key={`model2-risk-assessment:${dashboard.model2_development.status}`}
        featureNames={dashboard.model2_development_explainability.feature_names}
        available={developmentAvailable}
      />
      <MonitoringBatchRiskAssessment
        key={`monitoring-batch-risk-assessment:${dashboard.model2_development.status}`}
        refreshKey={refreshKey}
        available={developmentAvailable}
      />
      <DashboardPanel
        title="DEVELOPMENT Model 2 Explainability"
        subtitle="Synthetic experiment only · not PRODUCTION model explanation"
      >
        <div className="report-status-banner warning">
          <AlertTriangle size={20} />
          <span>
            <strong>SYNTHETIC DEVELOPMENT ONLY</strong>
            <small>
              {dashboard.model2_development_explainability.model_version
                ? `Model ${dashboard.model2_development_explainability.model_version}`
                : "Development artifact unavailable"}
            </small>
          </span>
        </div>
        {developmentAvailable &&
        dashboard.model2_development_explainability.status === "AVAILABLE" ? (
          <>
            <div style={{ width: "100%", height: 370 }}>
              <ResponsiveContainer>
                <BarChart
                  data={[...dashboard.model2_development_explainability.features].reverse()}
                  layout="vertical"
                  margin={{ top: 8, right: 48, bottom: 24, left: 8 }}
                >
                  <CartesianGrid stroke="#20324c" strokeDasharray="3 3" horizontal={false} />
                  <XAxis
                    type="number"
                    domain={[0, "dataMax"]}
                    tick={{ fill: "#a9bad1", fontSize: 10 }}
                    label={{
                      value: "Mean decrease in impurity importance",
                      position: "insideBottom",
                      offset: -14,
                      fill: "#a9bad1",
                      fontSize: 10,
                    }}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    width={190}
                    tickFormatter={(value: string) => value.replaceAll("_", " ")}
                    tick={{ fill: "#dce7f5", fontSize: 10 }}
                  />
                  <Tooltip
                    formatter={(value) => [Number(value).toFixed(4), "Importance"]}
                    contentStyle={{
                      background: "#0b1730",
                      border: "1px solid #224474",
                      borderRadius: 9,
                      color: "#eff7ff",
                    }}
                  />
                  <RechartsBar
                    dataKey="importance"
                    name="Importance"
                    fill="#22c6a8"
                    radius={[0, 4, 4, 0]}
                    isAnimationActive={false}
                  >
                    <LabelList
                      dataKey="importance"
                      position="right"
                      formatter={(value) => Number(value).toFixed(4)}
                      fill="#dce7f5"
                      fontSize={9}
                    />
                  </RechartsBar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="detail-list">
              <DetailRow
                label="Model version"
                value={dashboard.model2_development_explainability.model_version ?? "—"}
              />
              <DetailRow
                label="Provenance"
                value={dashboard.model2_development_explainability.provenance}
              />
              <DetailRow
                label="Generated"
                value={formatDateTime(
                  dashboard.model2_development_explainability.generation_timestamp,
                )}
              />
            </div>
            <div className="health-check-list">
              {dashboard.model2_development_explainability.features.map((feature) => (
                <div
                  className="health-check-row"
                  key={feature.feature}
                  style={{
                    gridTemplateColumns: "minmax(0, 1fr) auto",
                    padding: "10px 0",
                  }}
                >
                  <span>{feature.rank}. {feature.feature.replaceAll("_", " ")}</span>
                  <strong>{feature.importance.toFixed(4)}</strong>
                  <small style={{ gridColumn: "1 / -1", color: "#9aacc8" }}>
                    {feature.measures} {feature.model_rationale} {feature.risk_relation}
                  </small>
                </div>
              ))}
            </div>
            <p className="data-note">
              {dashboard.model2_development_explainability.causality_warning}
            </p>
          </>
        ) : (
          <p className="data-note">
            {dashboard.model2_development_explainability.reason ??
              "Generate the development-only feature-importance report to display this section."}
          </p>
        )}
      </DashboardPanel>
      <section className="two-column-page-grid">
        <DashboardPanel title="System Health" subtitle="Current backend-reported readiness">
          <div className="health-check-list">
            {systemRows.map((row) => (
              <div className="health-check-row" key={row.label}>
                <span className={`health-dot ${row.state}`} />
                <span>{row.label}</span>
                <strong className={row.state}>{row.value.replaceAll("_", " ")}</strong>
              </div>
            ))}
          </div>
          {dashboard.model2.status !== "TRAINED" && (
            <p className="data-note">
              Model 2: {dashboard.model2.labeled_observation_count}/
              {dashboard.model2.minimum_labeled_observations} labeled production monitoring batches.
              {dashboard.model2.reason ? ` ${dashboard.model2.reason}` : ""}
            </p>
          )}
        </DashboardPanel>
        <DashboardPanel title="Latest Monitoring Run" subtitle="Batch-level health and data quality">
          {dashboard.latest_monitoring ? (
            <>
              <div className={`report-status-banner ${dashboard.latest_monitoring.status.toLowerCase()}`}>
                <Activity size={20} />
                <span><strong>{dashboard.latest_monitoring.status}</strong><small>{dashboard.latest_monitoring.source.replaceAll("_", " ")}</small></span>
              </div>
              <div className="detail-list">
                <DetailRow label="Batch ID" value={dashboard.latest_monitoring.batch_id} />
                <DetailRow label="Timestamp" value={formatDateTime(dashboard.latest_monitoring.timestamp)} />
                <DetailRow label="Quality status" value={dashboard.latest_monitoring.quality_status} />
                <DetailRow label="Rows" value={formatNumber(dashboard.latest_monitoring.rows)} />
              </div>
            </>
          ) : <EmptyState text="No monitoring run is available yet." />}
        </DashboardPanel>
        <DashboardPanel title="Data Quality" subtitle={`Latest batch · ${formatNumber(dashboard.quality.total_rows)} rows`}>
          <div className="quality-bar-list">
            <QualityBar label="Missing values" value={dashboard.quality.missing_percentage} />
            <QualityBar label="Duplicate records" value={dashboard.quality.duplicate_percentage} />
            <QualityBar label="Invalid values" value={dashboard.quality.invalid_percentage} />
          </div>
        </DashboardPanel>
        <DashboardPanel title="Monitoring Coverage" subtitle="Selected dashboard period">
          <div className="coverage-summary">
            <strong>{formatNumber(dashboard.monitoring.report_count)}</strong>
            <span>reports marked real in the last {dashboard.monitoring.period_days} days</span>
            <strong>{formatNumber(dashboard.monitoring.batch_count)}</strong>
            <span>distinct real-source batches</span>
            <strong>{formatNumber(dashboard.monitoring.simulation_report_count)}</strong>
            <span>persisted simulation reports based on the original hotel booking dataset</span>
            <strong>{formatNumber(dashboard.monitoring.simulation_record_count)}</strong>
            <span>records represented across those simulated batches (not production traffic)</span>
            <strong>{formatNumber(dashboard.monitoring.source_counts.unknown)}</strong>
            <span>reports with unknown provenance, excluded from live outcomes</span>
          </div>
        </DashboardPanel>
        <DashboardPanel title="Data Provenance" subtitle="Monitoring reports grouped by declared source">
          <div className="health-check-list">
            <div className="health-check-row"><span className="health-dot good" /><span>Operator-declared real reports</span><strong>{formatNumber(dashboard.monitoring.source_counts.real)}</strong></div>
            <div className="health-check-row"><span className="health-dot pending" /><span>Original-dataset simulations</span><strong>{formatNumber(dashboard.monitoring.simulation_report_count)}</strong></div>
            <div className="health-check-row"><span className="health-dot pending" /><span>Unknown provenance reports</span><strong>{formatNumber(dashboard.monitoring.source_counts.unknown)}</strong></div>
          </div>
          <p className="data-note">Simulated batches remain visibly separate from real traffic. Only trusted production observations and matched, received outcomes contribute to operational metrics or Model 2 training.</p>
        </DashboardPanel>
      </section>
      <DashboardPanel
        title="Recent Model 1 Prediction History"
        subtitle="Production observations only; unlabeled outcomes remain pending"
      >
        {dashboard.production_observations.prediction_history.length ? (
          <div className="alert-history-table-wrap">
            <table className="alert-history-table">
              <thead>
                <tr>
                  <th>Observation ID</th>
                  <th>Batch</th>
                  <th>Predicted</th>
                  <th>Cancellation probability</th>
                  <th>Actual outcome</th>
                  <th>Predicted at</th>
                </tr>
              </thead>
              <tbody>
                {dashboard.production_observations.prediction_history.map((item) => (
                  <tr key={item.observation_id}>
                    <td>{item.observation_id}</td>
                    <td>{item.batch_id}</td>
                    <td>{item.predicted_class === 1 ? "Canceled" : "Not canceled"}</td>
                    <td>{item.probability_positive == null ? "—" : item.probability_positive.toFixed(3)}</td>
                    <td>{item.actual_outcome == null ? "Pending" : item.actual_outcome === 1 ? "Canceled" : "Not canceled"}</td>
                    <td>{formatDateTime(item.timestamp)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState text="No trusted production observations have been ingested yet." />
        )}
      </DashboardPanel>
    </>
  );
}

function MonitoringCsvUpload({ onRefresh }: { onRefresh: () => void }) {
  const [batchId, setBatchId] = useState("");
  const [source, setSource] = useState<"unknown" | "real" | "synthetic_test">("unknown");
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setMessage("");
    setError("");
    if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(batchId.trim())) {
      setError("Use a batch ID with letters, numbers, dots, underscores or hyphens (maximum 128 characters).");
      return;
    }
    if (!file || file.size > 10 * 1024 * 1024) {
      setError("Choose a CSV file no larger than 10 MB.");
      return;
    }
    setSubmitting(true);
    try {
      const params = new URLSearchParams({
        batch_id: batchId.trim(),
        data_source_type: source,
      });
      const response = await apiFetch(`${API}/api/v1/monitoring/batches/csv?${params}`, {
        method: "POST",
        headers: { "Content-Type": "text/csv" },
        body: file,
      });
      const result = await response.json() as {
        batch_id?: string;
        status?: string;
        quality_status?: string;
        detail?: string;
      };
      if (!response.ok) {
        throw new Error(result.detail ?? `Monitoring run failed (HTTP ${response.status}).`);
      }
      setMessage(`${result.status} · ${result.quality_status} · report saved for ${result.batch_id}.`);
      setBatchId("");
      setFile(null);
      onRefresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Could not run monitoring.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="workflow-upload-form" onSubmit={(event) => void submit(event)}>
      <label>Batch ID<input required maxLength={128} value={batchId} onChange={(event) => setBatchId(event.target.value)} placeholder="booking-batch-2026-11" /></label>
      <label>Provenance<select value={source} onChange={(event) => setSource(event.target.value as typeof source)}><option value="unknown">Unknown</option><option value="real">Real (operator-declared)</option><option value="synthetic_test">Simulated</option></select></label>
      <label>Monitoring CSV<input required type="file" accept=".csv,text/csv" onChange={(event) => setFile(event.target.files?.[0] ?? null)} /></label>
      <button type="submit" className="outline-action" disabled={submitting}>{submitting ? "Monitoring…" : "Run monitoring"}</button>
      {message && <p className="workflow-success" role="status">{message}</p>}
      {error && <p className="alert-error" role="alert">{error}</p>}
      <small>Provenance is operator-declared, not independently authenticated. Reports marked real are eligible for genuine outcome evaluation.</small>
    </form>
  );
}

type DevelopmentPrediction = {
  api_status: string;
  model_name: string;
  model_version: string;
  model_provenance: string;
  prediction: number;
  predicted_class_label: string;
  class_probabilities: Record<string, number>;
  risk_assessment: {
    predicted_class: string;
    degradation_probability: number;
    risk_level: string | null;
  };
  explanation: string;
  prediction_timestamp: string;
};

type MonitoringBatchPrediction = DevelopmentPrediction & {
  batch_id: string;
  unavailable_features: string[];
  production_validated: false;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isFiniteNumberRecord(value: unknown): value is Record<string, number> {
  return (
    isObject(value) &&
    Object.values(value).every(
      (item) => typeof item === "number" && Number.isFinite(item),
    )
  );
}

function isOfflineModel2Summary(
  value: unknown,
): value is DashboardData["model2_offline"] {
  if (!isObject(value)) return false;
  const numericFields = [
    "historical_observation_count",
    "eligible_observation_count",
    "training_size",
    "testing_size",
  ];
  const metrics = value.evaluation_metrics;
  const metricsValid =
    metrics === null ||
    (isObject(metrics) &&
      ["accuracy", "precision", "recall", "f1_score"].every(
        (name) =>
          typeof metrics[name] === "number" &&
          Number.isFinite(metrics[name]),
      ) &&
      (metrics.roc_auc === null ||
        (typeof metrics.roc_auc === "number" &&
          Number.isFinite(metrics.roc_auc))) &&
      Array.isArray(metrics.confusion_matrix) &&
      isFiniteNumberRecord(metrics.test_class_counts));
  return (
    (value.status === "TRAINED" || value.status === "NOT_TRAINED") &&
    value.model_scope === "HISTORICAL_OFFLINE" &&
    value.provenance === "HISTORICAL_OFFLINE" &&
    (value.status === "TRAINED"
      ? typeof value.model1_replay_version === "string" &&
        value.model1_replay_version.trim().length > 0
      : value.model1_replay_version === null ||
        (typeof value.model1_replay_version === "string" &&
          value.model1_replay_version.trim().length > 0)) &&
    numericFields.every(
      (name) =>
        typeof value[name] === "number" &&
        Number.isFinite(value[name]),
    ) &&
    isFiniteNumberRecord(value.target_class_counts) &&
    metricsValid &&
    (value.confusion_matrix === null ||
      (Array.isArray(value.confusion_matrix) &&
        value.confusion_matrix.every(
          (row) =>
            Array.isArray(row) &&
            row.every(
              (item) =>
                typeof item === "number" && Number.isFinite(item),
            ),
        ))) &&
    Array.isArray(value.evaluation_predictions) &&
    Array.isArray(value.observation_history) &&
    Array.isArray(value.missing_requirements) &&
    value.missing_requirements.every((item) => typeof item === "string")
  );
}

function isModel2Summary(value: unknown): value is DashboardData["model2"] {
  if (!isObject(value)) return false;
  const numericFields = [
    "production_batch_count",
    "genuine_production_observation_count",
    "labeled_batch_count",
    "eligible_labeled_observation_count",
    "unlabeled_batch_count",
    "labeled_observation_count",
    "training_dataset_size",
    "minimum_labeled_observations",
    "training_progress_percent",
    "training_size",
    "testing_size",
    "evaluation_dataset_size",
  ];
  const metrics = value.evaluation_metrics;
  const metricsValid =
    metrics === null ||
    (isObject(metrics) &&
      ["accuracy", "precision", "recall", "f1_score"].every(
        (name) =>
          typeof metrics[name] === "number" &&
          Number.isFinite(metrics[name]),
      ) &&
      (metrics.roc_auc === null ||
        (typeof metrics.roc_auc === "number" &&
          Number.isFinite(metrics.roc_auc))) &&
      (metrics.pr_auc === null ||
        (typeof metrics.pr_auc === "number" &&
          Number.isFinite(metrics.pr_auc))) &&
      Array.isArray(metrics.confusion_matrix) &&
      metrics.confusion_matrix.every(
        (row) =>
          Array.isArray(row) &&
          row.every(
            (item) => typeof item === "number" && Number.isFinite(item),
          ),
      ) &&
      isFiniteNumberRecord(metrics.test_class_counts));
  const predictionsValid =
    Array.isArray(value.prediction_history) &&
    value.prediction_history.every(
      (prediction) =>
        isObject(prediction) &&
        typeof prediction.prediction_id === "string" &&
        typeof prediction.batch_id === "string" &&
        typeof prediction.prediction_timestamp === "string" &&
        Number.isFinite(Date.parse(prediction.prediction_timestamp)) &&
        prediction.source === "production" &&
        prediction.data_source_type === "real" &&
        typeof prediction.model_version === "string" &&
        (prediction.predicted_degradation_risk === 0 ||
          prediction.predicted_degradation_risk === 1) &&
        typeof prediction.degradation_probability === "number" &&
        Number.isFinite(prediction.degradation_probability) &&
        prediction.degradation_probability >= 0 &&
        prediction.degradation_probability <= 1 &&
        ["LOW", "MEDIUM", "HIGH"].includes(
          String(prediction.risk_category),
        ) &&
        ["HEALTHY", "WARNING", "CRITICAL"].includes(
          String(prediction.health_status),
        ),
    );

  return (
    ["TRAINED", "NOT_TRAINED"].includes(String(value.status)) &&
    typeof value.training_eligibility === "string" &&
    (value.reason === null || typeof value.reason === "string") &&
    Array.isArray(value.missing_requirements) &&
    value.missing_requirements.every((item) => typeof item === "string") &&
    numericFields.every(
      (name) =>
        typeof value[name] === "number" && Number.isFinite(value[name]),
    ) &&
    isFiniteNumberRecord(value.target_class_counts) &&
    isFiniteNumberRecord(value.training_class_counts) &&
    isFiniteNumberRecord(value.testing_class_counts) &&
    metricsValid &&
    predictionsValid
  );
}

function isMonitoringBatchPrediction(
  value: unknown,
): value is MonitoringBatchPrediction {
  if (!isObject(value) || !isObject(value.class_probabilities)) return false;
  const stableProbability = value.class_probabilities.stable;
  const degradedProbability = value.class_probabilities.degraded;
  const riskAssessment = value.risk_assessment;
  return (
    typeof value.batch_id === "string" &&
    value.api_status === "DEVELOPMENT_PREDICTION" &&
    typeof value.model_name === "string" &&
    typeof value.model_version === "string" &&
    value.model_provenance === "SYNTHETIC_DEVELOPMENT_ONLY" &&
    (value.prediction === 0 || value.prediction === 1) &&
    (value.predicted_class_label === "stable" ||
      value.predicted_class_label === "degraded") &&
    ((value.prediction === 0 && value.predicted_class_label === "stable") ||
      (value.prediction === 1 && value.predicted_class_label === "degraded")) &&
    Object.keys(value.class_probabilities).length === 2 &&
    typeof stableProbability === "number" &&
    Number.isFinite(stableProbability) &&
    stableProbability >= 0 &&
    stableProbability <= 1 &&
    typeof degradedProbability === "number" &&
    Number.isFinite(degradedProbability) &&
    degradedProbability >= 0 &&
    degradedProbability <= 1 &&
    Math.abs(stableProbability + degradedProbability - 1) <= 0.01 &&
    Array.isArray(value.unavailable_features) &&
    value.unavailable_features.every((feature) => typeof feature === "string") &&
    value.production_validated === false &&
    isObject(riskAssessment) &&
    typeof riskAssessment.predicted_class === "string" &&
    riskAssessment.predicted_class === value.predicted_class_label &&
    typeof riskAssessment.degradation_probability === "number" &&
    Number.isFinite(riskAssessment.degradation_probability) &&
    riskAssessment.degradation_probability === degradedProbability &&
    typeof value.explanation === "string" &&
    typeof value.prediction_timestamp === "string"
  );
}

export function MonitoringBatchRiskAssessment({
  refreshKey,
  available,
}: {
  refreshKey: number;
  available: boolean;
}) {
  const [reports, setReports] = useState<MonitoringReportSummary[]>([]);
  const [selectedBatchId, setSelectedBatchId] = useState("");
  const [loadedRefreshKey, setLoadedRefreshKey] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [prediction, setPrediction] = useState<MonitoringBatchPrediction | null>(null);
  const loadingReports = loadedRefreshKey !== refreshKey;

  useEffect(() => {
    const controller = new AbortController();
    if (!available) {
      return () => controller.abort();
    }
    apiFetch(
      `${API}/api/v1/monitoring/reports?${new URLSearchParams({
        period_days: "3650",
        limit: "1000",
        source: "SIMULATED",
      })}`,
      { signal: controller.signal },
    )
      .then(async (response) => {
        const result: unknown = await response.json().catch(() => null);
        if (!response.ok) {
          const detail = isObject(result) && typeof result.detail === "string"
            ? result.detail
            : `Monitoring batch list failed (HTTP ${response.status}).`;
          throw new Error(detail);
        }
        if (!isObject(result) || !Array.isArray(result.reports)) {
          throw new Error("Monitoring batch list returned an invalid response.");
        }
        const summaries = result.reports.filter(
          (report): report is MonitoringReportSummary =>
            isObject(report) &&
            typeof report.batch_id === "string" &&
            typeof report.timestamp === "string" &&
            typeof report.status === "string" &&
            typeof report.quality_status === "string" &&
            report.source === "SIMULATED" &&
            typeof report.rows === "number" &&
            Number.isFinite(report.rows) &&
            typeof report.download_url === "string",
        );
        const seenBatchIds = new Set<string>();
        setReports(
          summaries.filter((report) => {
            if (seenBatchIds.has(report.batch_id)) return false;
            seenBatchIds.add(report.batch_id);
            return true;
          }),
        );
        setError("");
      })
      .catch((requestError: unknown) => {
        if (requestError instanceof DOMException && requestError.name === "AbortError") return;
        setReports([]);
        setError(
          requestError instanceof Error
            ? requestError.message
            : "Could not load monitoring batches.",
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadedRefreshKey(refreshKey);
      });
    return () => controller.abort();
  }, [available, refreshKey]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!available) return;
    if (!selectedBatchId) {
      setError("Select a monitoring batch before requesting a prediction.");
      return;
    }

    setError("");
    setPrediction(null);
    setSubmitting(true);
    try {
      const response = await apiFetch(`${API}/api/v1/model2/development/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_id: selectedBatchId }),
      });
      const result: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        const detail = isObject(result) && typeof result.detail === "string"
          ? result.detail
          : `Development prediction failed (HTTP ${response.status}).`;
        throw new Error(detail);
      }
      if (!isMonitoringBatchPrediction(result)) {
        throw new Error("Development prediction returned an invalid response.");
      }
      setPrediction(result);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not request a development prediction.",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <DashboardPanel
      title="DEVELOPMENT monitoring batch risk assessment"
      subtitle="Uses SIMULATED monitoring batches only; no production observations enter this development prediction"
    >
      <div className="report-status-banner warning">
        <AlertTriangle size={20} />
        <span>
          <strong>DEVELOPMENT only — trained on synthetic data, not production validated. Do not use for production decisions.</strong>
        </span>
      </div>
      {!available ? (
        <p className="data-note" role="status">
          The DEVELOPMENT model is unavailable. Batch predictions are disabled.
        </p>
      ) : (
      <form className="workflow-upload-form" onSubmit={(event) => void submit(event)}>
        <label>
          DEVELOPMENT simulated monitoring batch
          <select
            value={selectedBatchId}
            onChange={(event) => {
              setSelectedBatchId(event.target.value);
              setPrediction(null);
              setError("");
            }}
            disabled={loadingReports || reports.length === 0}
            required
          >
            <option value="">
              {loadingReports
                ? "Loading monitoring batches…"
                : reports.length
                  ? "Select a monitoring batch"
                  : "No monitoring batches available"}
            </option>
            {reports.map((report) => (
              <option
                key={`${report.batch_id}:${report.timestamp}`}
                value={report.batch_id}
              >
                {report.source} · {report.batch_id} · {report.status} · {formatDateTime(report.timestamp)}
              </option>
            ))}
          </select>
        </label>
        <button
          type="submit"
          className="outline-action"
          disabled={!available || loadingReports || submitting || !selectedBatchId}
        >
          {submitting ? "Assessing batch…" : "Assess selected batch"}
        </button>
      </form>
      )}
      {available && loadingReports && <p className="data-note" role="status">Loading SIMULATED monitoring batches…</p>}
      {available && !loadingReports && !error && reports.length === 0 && (
        <p className="data-note">No existing monitoring batches are available for assessment.</p>
      )}
      {available && error && <p className="alert-error" role="alert">{error}</p>}
      {available && prediction && (
        <>
          <div className="detail-list">
            <DetailRow label="Batch ID" value={prediction.batch_id} />
            <DetailRow label="Predicted class" value={prediction.predicted_class_label} />
            <DetailRow
              label="Prediction status"
              value={`${prediction.api_status} · NOT PRODUCTION VALIDATED`}
            />
            <DetailRow label="Prediction timestamp" value={formatDateTime(prediction.prediction_timestamp)} />
          </div>
          <div className="detail-list">
            {Object.entries(prediction.class_probabilities).map(([label, probability]) => (
              <DetailRow
                key={label}
                label={`${label} probability`}
                value={`${(probability * 100).toFixed(1)}%`}
              />
            ))}
          </div>
          <div className="detail-list">
            <DetailRow
              label="Unavailable features"
              value={prediction.unavailable_features.length
                ? prediction.unavailable_features.map((feature) => feature.replaceAll("_", " ")).join(", ")
                : "None"}
            />
          </div>
        </>
      )}
    </DashboardPanel>
  );
}

export function Model2DevelopmentRiskAssessment({
  featureNames,
  available,
}: {
  featureNames: string[];
  available: boolean;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [prediction, setPrediction] = useState<DevelopmentPrediction | null>(null);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [unavailable, setUnavailable] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!available) return;
    setError("");
    setPrediction(null);
    setUnavailable(false);

    const features: Record<string, number> = {};
    for (const name of featureNames) {
      const rawValue = values[name];
      const numericValue = rawValue === undefined || rawValue.trim() === ""
        ? Number.NaN
        : Number(rawValue);
      if (!Number.isFinite(numericValue)) {
        setError(`Enter a finite numeric value for ${name.replaceAll("_", " ")}.`);
        return;
      }
      features[name] = numericValue;
    }

    setSubmitting(true);
    try {
      const response = await apiFetch(`${API}/api/v1/model2/development/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ monitoring_features: features }),
      });
      const result = await response.json() as DevelopmentPrediction & { detail?: string };
      if (!response.ok) {
        setUnavailable(response.status === 503);
        throw new Error(result.detail ?? `Development prediction failed (HTTP ${response.status}).`);
      }
      setPrediction(result);
    } catch (requestError) {
      setError(requestError instanceof Error
        ? requestError.message
        : "Could not request a development prediction.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <DashboardPanel
      title="Model 2 Development Risk Assessment"
      subtitle="DEVELOPMENT only · separate synthetic-development prediction · never production health status"
    >
      <div className="report-status-banner warning">
        <AlertTriangle size={20} />
        <span>
          <strong>SYNTHETIC DEVELOPMENT ONLY</strong>
          <small>Enter an observation manually; the form does not generate or fill synthetic observations.</small>
        </span>
      </div>
      {!available ? (
        <p className="data-note" role="status">
          The DEVELOPMENT model is unavailable. Manual predictions are disabled.
        </p>
      ) : featureNames.length > 0 ? (
        <form className="workflow-upload-form" onSubmit={(event) => void submit(event)}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
            {featureNames.map((name) => (
              <label key={name}>
                {name.replaceAll("_", " ")}
                <input
                  type="number"
                  step="any"
                  required
                  value={values[name] ?? ""}
                  onChange={(event) => setValues((current) => ({
                    ...current,
                    [name]: event.target.value,
                  }))}
                />
              </label>
            ))}
          </div>
          <button type="submit" className="outline-action" disabled={!available || submitting}>
            {submitting ? "Assessing…" : "Run development assessment"}
          </button>
        </form>
      ) : (
        <p className="data-note">Model 2 feature schema is unavailable; a development prediction cannot be submitted.</p>
      )}
      {available && unavailable && (
        <p className="data-note" role="status">Development model or metadata is unavailable. No production model fallback is used.</p>
      )}
      {available && error && <p className="alert-error" role="alert">{error}</p>}
      {available && prediction && (
        <>
          <div className="detail-list">
            <DetailRow label="Predicted class" value={prediction.predicted_class_label} />
            <DetailRow label="Degraded-class probability" value={`${(prediction.risk_assessment.degradation_probability * 100).toFixed(1)}%`} />
            <DetailRow label="Model version" value={prediction.model_version} />
            <DetailRow label="Provenance" value={prediction.model_provenance} />
            <DetailRow label="Prediction timestamp" value={formatDateTime(prediction.prediction_timestamp)} />
          </div>
          <div className="detail-list">
            {Object.entries(prediction.class_probabilities).map(([label, probability]) => (
              <DetailRow key={label} label={`${label} probability`} value={`${(probability * 100).toFixed(1)}%`} />
            ))}
          </div>
          <p className="data-note">{prediction.explanation}</p>
        </>
      )}
    </DashboardPanel>
  );
}

function DashboardPanel({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
}) {
  return (
    <article className="workspace-panel">
      <header className="workspace-panel-heading">
        <span className="panel-accent-icon"><Activity size={17} /></span>
        <span><strong>{title}</strong>{subtitle && <small>{subtitle}</small>}</span>
      </header>
      <div className="workspace-panel-content">{children}</div>
    </article>
  );
}

function MetricSummary({
  label,
  value,
  note,
  icon,
}: {
  label: string;
  value: string;
  note: string;
  icon: ReactNode;
}) {
  return (
    <article className="metric-card summary-metric">
      <span className="metric-card-label">{icon}{label}</span>
      <strong>{value}</strong>
      <small>{note}</small>
    </article>
  );
}

function ModelInformation({ dashboard }: { dashboard: DashboardData }) {
  return (
    <div className="detail-list">
      <DetailRow label="Model name" value={dashboard.model.name} />
      <DetailRow label="Algorithm" value={dashboard.model.algorithm} />
      <DetailRow label="Version" value={`v${dashboard.model.version}`} />
      <DetailRow label="Training date" value={formatDate(dashboard.model.training_date)} />
      <DetailRow label="Original data rows" value={formatNumber(dashboard.model.dataset_row_count)} />
      <DetailRow label="Arrival date coverage" value={dashboard.model.dataset_period} />
      <DetailRow label="Held-out test records" value={formatNumber(dashboard.metrics.sample_count)} />
      <DetailRow label="Status" value={dashboard.model.status.replaceAll("_", " ")} state={dashboard.model.status === "READY" ? "good" : "bad"} />
    </div>
  );
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  if (matrix.length !== 2 || matrix.some((row) => row.length !== 2)) {
    return <EmptyState text="Confusion matrix is unavailable." />;
  }
  return (
    <div className="confusion-matrix">
      <span />
      <span>Predicted<br />Not Cancel</span>
      <span>Predicted<br />Cancel</span>
      <strong>Actual<br />Not Cancel</strong>
      <b>{formatNumber(matrix[0][0])}</b>
      <b>{formatNumber(matrix[0][1])}</b>
      <strong>Actual<br />Cancel</strong>
      <b>{formatNumber(matrix[1][0])}</b>
      <b>{formatNumber(matrix[1][1])}</b>
    </div>
  );
}

function PredictionSummary({ dashboard }: { dashboard: DashboardData }) {
  return (
    <div className="prediction-summary">
      <strong>{formatNumber(dashboard.predictions.total)}</strong>
      <span>predictions from verified real batches</span>
      <div><small>Today</small><b>{formatNumber(dashboard.predictions.today)}</b></div>
      <div><small>Batches</small><b>{formatNumber(dashboard.predictions.batch_count)}</b></div>
      <div><small>Distribution records</small><b>{formatNumber(dashboard.predictions.distribution_sample_count)}</b></div>
    </div>
  );
}

function DetailRow({
  label,
  value,
  state,
}: {
  label: string;
  value: string;
  state?: string;
}) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong className={state}>{value || "—"}</strong>
    </div>
  );
}

function QualityBar({ label, value }: { label: string; value: number | null }) {
  const percentage = value == null ? null : Math.max(0, Math.min(100, value));
  return (
    <div className="quality-bar">
      <div><span>{label}</span><strong>{percentage == null ? "—" : `${percentage.toFixed(2)}%`}</strong></div>
      <i><b style={{ width: `${percentage ?? 0}%` }} /></i>
    </div>
  );
}

function formatDateTime(value: string | null | undefined) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "—"
    : new Intl.DateTimeFormat(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      }).format(date);
}

function PanelHeading({
  number,
  title,
  subtitle,
  right,
}: {
  number: string;
  title: string;
  subtitle?: string;
  right?: ReactNode;
}) {
  return (
    <div className="panel-heading">
      <span className="panel-number">{number}</span>
      <div className="panel-heading-copy">
        <h2>{title}</h2>
        {subtitle && <small>{subtitle}</small>}
      </div>
      {right && <div className="panel-heading-right">{right}</div>}
    </div>
  );
}

function MiniStat({
  color,
  icon,
  value,
  label,
}: {
  color: string;
  icon: ReactNode;
  value: string;
  label: string;
}) {
  return (
    <div className={`mini-stat ${color}`}>
      <span>{icon}</span>
      <strong>{value}</strong>
      <small>{label}</small>
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return <div className="empty-state"><CircleHelp size={17} /><span>{text}</span></div>;
}

function DataWorkflowGuide({
  title,
  description,
  steps,
  actionLabel,
  onAction,
}: {
  title: string;
  description: string;
  steps: string[];
  actionLabel?: string;
  onAction?: () => void;
}) {
  return (
    <div className="empty-workflow">
      <div className="empty-workflow-heading">
        <CircleHelp size={18} />
        <span>
          <strong>{title}</strong>
          <small>{description}</small>
        </span>
      </div>
      <ol>
        {steps.map((step, index) => (
          <li key={step}>
            <span>{index + 1}</span>
            <p>{step}</p>
          </li>
        ))}
      </ol>
      {actionLabel && onAction && (
        <button type="button" className="outline-action" onClick={onAction}>
          {actionLabel}
        </button>
      )}
    </div>
  );
}

function SystemRow({
  label,
  value,
  state,
}: {
  label: string;
  value: string;
  state: string;
}) {
  return (
    <div className="system-row">
      <i className={`system-light ${state}`} />
      <span>{label}</span>
      <strong className={state}>{value}</strong>
    </div>
  );
}

const REMEMBERED_USERNAME_STORAGE_KEY = "smart-ml-remembered-username";

function getRememberedUsername() {
  try {
    return window.localStorage.getItem(REMEMBERED_USERNAME_STORAGE_KEY) ?? "";
  } catch (error) {
    console.error("Could not load remembered username.", error);
    return "";
  }
}

function SignIn({
  onSignIn,
  initialError,
}: {
  onSignIn: (username: string) => void;
  initialError: string;
}) {
  const [username, setUsername] = useState(getRememberedUsername);
  const [password, setPassword] = useState("");
  const [rememberMe, setRememberMe] = useState(Boolean(username));
  const [passwordVisible, setPasswordVisible] = useState(false);
  const [message, setMessage] = useState(initialError);
  const [submitting, setSubmitting] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitting) return;
    setMessage("");
    setSubmitting(true);
    try {
      const response = await apiFetch(`${API}/api/v1/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const result = await response.json() as {
        username?: string;
        detail?: string;
      };
      if (!response.ok || !result.username) {
        throw new Error(result.detail ?? `Sign in failed (HTTP ${response.status}).`);
      }

      try {
        if (rememberMe) {
          window.localStorage.setItem(REMEMBERED_USERNAME_STORAGE_KEY, username);
        } else {
          window.localStorage.removeItem(REMEMBERED_USERNAME_STORAGE_KEY);
        }
      } catch (storageError) {
        console.error("Could not update remembered username.", storageError);
      }
      setPassword("");
      onSignIn(result.username);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Could not sign in.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="sign-in-page">
      <header className="sign-in-header">
        <a className="sign-in-brand" href="#home" aria-label="Smart ML home">
          <span className="sign-in-brand-mark">
            <HeartPulse size={31} strokeWidth={2} />
          </span>
          <span>
            <strong>Smart ML</strong>
            <small>Model Health Monitor</small>
          </span>
        </a>
        <nav className="sign-in-nav" aria-label="Main navigation">
          <a className="selected" href="#home">Home</a>
          <a href="#features">Features</a>
          <a href="#about">About</a>
        </nav>
      </header>

      <div className="sign-in-layout" id="home">
        <section className="sign-in-intro" id="about">
          <span className="intro-badge">
            <i />
            AI-Powered Model Reliability
          </span>
          <h1>
            Smart ML Model
            <span>Health Monitor</span>
          </h1>
          <p className="intro-description">
            Continuously monitor your ML models, detect data and performance
            drift, and get early warnings before it impacts your business.
          </p>

          <div className="feature-chips" id="features">
            <div><BarChart3 /><span>Performance<br />Tracking</span></div>
            <div><Activity /><span>Data Drift<br />Detection</span></div>
            <div><Bell /><span>Early<br />Warnings</span></div>
            <div><TrendingUp /><span>Degradation<br />Prediction</span></div>
          </div>

          <div className="sign-in-visual" aria-hidden="true">
            <div className="visual-orbit orbit-one" />
            <div className="visual-orbit orbit-two" />
            <div className="visual-nodes">
              <span>ML</span>
              <i /><i /><i /><i /><i /><i />
            </div>
            <div className="visual-laptop">
              <div className="laptop-screen">
                <div className="visual-screen-header">
                  <span><i />Model Health</span>
                  <small>PREVIEW</small>
                </div>
                <div className="visual-chart-area">
                  <div className="visual-gauge"><strong>—</strong><small>Illustrative</small></div>
                  <svg viewBox="0 0 220 92" preserveAspectRatio="none">
                    <defs>
                      <linearGradient id="visual-line" x1="0" x2="1">
                        <stop stopColor="#00e8ed" />
                        <stop offset="1" stopColor="#d24cff" />
                      </linearGradient>
                    </defs>
                    <path d="M2 68 35 54 68 58 99 32 133 43 165 22 196 31 218 9" />
                  </svg>
                </div>
                <div className="visual-bars"><i /><i /><i /><i /><i /><i /><i /><i /></div>
              </div>
              <div className="laptop-base" />
            </div>
          </div>
        </section>

        <section className="sign-in-card" aria-labelledby="sign-in-title">
          <div className="sign-in-card-icon"><HeartPulse size={46} /></div>
          <h2 id="sign-in-title">Sign In</h2>
          <p className="sign-in-subtitle">Access your Smart ML Model Health Monitor</p>

          <form onSubmit={submit}>
            <label className="sign-in-input">
              <Mail size={21} />
              <input
                type="text"
                name="username"
                aria-label="Local username"
                autoComplete="username"
                placeholder="Enter your local username"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                disabled={submitting}
                required
              />
            </label>
            <label className="sign-in-input">
              <LockKeyhole size={21} />
              <input
                type={passwordVisible ? "text" : "password"}
                name="password"
                aria-label="Password"
                autoComplete="current-password"
                placeholder="Enter your password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                disabled={submitting}
                required
              />
              <button
                type="button"
                className="password-visibility"
                onClick={() => setPasswordVisible((visible) => !visible)}
                aria-label={passwordVisible ? "Hide password" : "Show password"}
              >
                {passwordVisible ? <EyeOff size={20} /> : <Eye size={20} />}
              </button>
            </label>
            <div className="sign-in-options">
              <label className="remember-option">
                <input
                  type="checkbox"
                  checked={rememberMe}
                  onChange={(event) => setRememberMe(event.target.checked)}
                />
                <span>Remember username</span>
              </label>
              <span className="sign-in-help">Password resets are not configured locally.</span>
            </div>
            <button type="submit" className="sign-in-submit" disabled={submitting}>
              {submitting ? "Signing in…" : "Sign In"} <ArrowRight size={22} />
            </button>
          </form>

          <p className="sign-in-signup">
            Local credentials are configured in the backend environment.
          </p>
          <p className="demo-notice">
            Local development authentication is enabled. Use the credentials configured for the backend; OIDC is a future deployment option.
          </p>
          {message && (
            <p className="sign-in-message" role="alert">
              {message}
              <button type="button" onClick={() => setMessage("")} aria-label="Dismiss message">
                <X size={14} />
              </button>
            </p>
          )}
        </section>
      </div>
    </main>
  );
}

function App() {
  const [userEmail, setUserEmail] = useState("");
  const [authChecking, setAuthChecking] = useState(true);
  const [authError, setAuthError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    async function restoreSession() {
      try {
        const response = await apiFetch(`${API}/api/v1/auth/session`, {
          signal: controller.signal,
        });
        if (response.status === 401) return;
        if (!response.ok) {
          throw new Error(`Authentication service returned HTTP ${response.status}.`);
        }
        const session = await response.json() as { username: string };
        if (!session.username) {
          throw new Error("Authentication service returned an invalid session.");
        }
        if (active) setUserEmail(session.username);
      } catch (error) {
        if (active && !(error instanceof DOMException && error.name === "AbortError")) {
          setAuthError(error instanceof Error ? error.message : "Could not restore the local session.");
        }
      } finally {
        if (active) setAuthChecking(false);
      }
    }

    void restoreSession();
    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  useEffect(() => {
    function expireSession() {
      setUserEmail("");
      setAuthError("Your local session expired. Sign in again.");
    }
    window.addEventListener("smart-ml-auth-expired", expireSession);
    return () => window.removeEventListener("smart-ml-auth-expired", expireSession);
  }, []);

  async function signOut() {
    const response = await apiFetch(`${API}/api/v1/auth/logout`, { method: "POST" });
    if (response.status === 401) {
      setAuthError("Your local session expired. Sign in again.");
      setUserEmail("");
      return;
    }
    if (!response.ok) {
      throw new Error(`Sign out failed (HTTP ${response.status}).`);
    }
    setAuthError("");
    setUserEmail("");
  }

  if (authChecking) {
    return <main className="sign-in-page"><p role="status">Checking local session…</p></main>;
  }

  if (!userEmail) {
    return <SignIn onSignIn={setUserEmail} initialError={authError} />;
  }

  return (
    <Dashboard
      userEmail={userEmail}
      onSignOut={signOut}
    />
  );
}

export default App;
