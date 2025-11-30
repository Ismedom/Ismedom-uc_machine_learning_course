import apiClient from "./client";

export const predictionApi = {
  predictSingle: (data: {
    hour: number;
    dayofweek: number;
    lag_1h: number;
    lag_24h: number;
    rolling_mean_24h: number;
  }) => apiClient.post("/api/predict_single", data),

  predictCost: (data: {
    price_per_kwh: number;
    peak_hours: [number, number];
    peak_multiplier: number;
    thresholds: {
      hourly_kW: number;
      daily_kWh: number;
      weekly_kWh: number;
      monthly_kWh: number;
    };
    days_ahead: number;
    current_prediction?: number;
  }) => apiClient.post("/api/cost/predict", data),
};

export default {
  prediction: predictionApi,
};
