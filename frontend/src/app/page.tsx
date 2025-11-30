"use client";

import { useState } from "react";
import { predictionApi } from "@/lib/api/endpoints";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  SparklesIcon,
  ExclamationTriangleIcon,
  CurrencyDollarIcon,
  ChartBarIcon,
} from "@heroicons/react/24/outline";

interface PredictionResult {
  prediction: number;
  features_used: {
    dayofweek: number;
    hour: number;
    lag_1h: number;
    lag_24h: number;
    rolling_mean_24h: number;
  };
}

interface CostAnalysisResult {
  projections: {
    days_predicted: number;
    daily_avg_cost: number;
    monthly_total_cost: number;
    yearly_total_cost: number;
    monthly_avg_kwh: number;
    yearly_total_kwh: number;
    peak_hours_daily: number;
    off_peak_hours_daily: number;
  };
  aggregated: {
    [key: string]: any[];
  };
  violations: {
    [key: string]: any[];
  };
  summary: {
    total_predicted_cost: number;
    total_predicted_kwh: number;
    avg_daily_cost: number;
    total_violations: number;
  };
}

interface CostSettings {
  price_per_kwh: number;
  peak_hours: [number, number];
  peak_multiplier: number;
  thresholds: {
    hourly_kW: number;
    daily_kWh: number;
    weekly_kWh: number;
    monthly_kWh: number;
  };
}

interface ApiResponse<T = any> {
  success: boolean;
  message?: string;
  data?: T;
  error?: string;
}

const daysOfWeek = [
  { value: 0, label: "ច័ន្ទ (Monday)" },
  { value: 1, label: "អង្គារ (Tuesday)" },
  { value: 2, label: "ពុធ (Wednesday)" },
  { value: 3, label: "ព្រហស្បតិ៍ (Thursday)" },
  { value: 4, label: "សុក្រ (Friday)" },
  { value: 5, label: "សៅរ៍ (Saturday)" },
  { value: 6, label: "អាទិត្យ (Sunday)" },
];

export default function Home() {
  const [activeTab, setActiveTab] = useState<"prediction" | "cost">(
    "prediction"
  );

  const [dayOfWeek, setDayOfWeek] = useState<number>(4);
  const [hour, setHour] = useState<string>("20");
  const [lag1h, setLag1h] = useState<string>("2.5");
  const [lag24h, setLag24h] = useState<string>("2.8");
  const [rollingMean24h, setRollingMean24h] = useState<string>("2.3");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);

  const [costSettings, setCostSettings] = useState<CostSettings>({
    price_per_kwh: 0.2,
    peak_hours: [18, 22],
    peak_multiplier: 1.5,
    thresholds: {
      hourly_kW: 5.0,
      daily_kWh: 50.0,
      weekly_kWh: 300.0,
      monthly_kWh: 1200.0,
    },
  });
  const [daysAhead, setDaysAhead] = useState<number | string>(30);
  const [costAnalysis, setCostAnalysis] = useState<CostAnalysisResult | null>(
    null
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = (await predictionApi.predictSingle({
        hour: parseFloat(hour),
        dayofweek: dayOfWeek,
        lag_1h: parseFloat(lag1h),
        lag_24h: parseFloat(lag24h),
        rolling_mean_24h: parseFloat(rollingMean24h),
      })) as unknown as ApiResponse<PredictionResult>;

      console.log("API Response:", response);

      if (response.success && response.data) {
        setPrediction(response.data);
      } else {
        setError(response.error || "Prediction failed");
      }
    } catch (err) {
      setError("Failed to connect to API. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleCostAnalysis = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = (await predictionApi.predictCost({
        price_per_kwh: costSettings.price_per_kwh,
        peak_hours: costSettings.peak_hours,
        peak_multiplier: costSettings.peak_multiplier,
        thresholds: costSettings.thresholds,
        days_ahead: Number(daysAhead),
        current_prediction: prediction?.prediction,
      })) as unknown as ApiResponse<CostAnalysisResult>;

      if (response.success && response.data) {
        setCostAnalysis(response.data);
      } else {
        setError(response.error || "Cost prediction failed");
      }
    } catch (err) {
      setError(
        "Failed to perform cost prediction. Make sure the backend is running and model is trained."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleChangeDaysAhead = (e: React.ChangeEvent<HTMLInputElement>) => {
    setDaysAhead(parseInt(e.target.value) || "");
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <Card className="p-8">
          <CardHeader className="text-center">
            <CardTitle className="text-3xl mb-2 flex items-center justify-center gap-2">
              <SparklesIcon className="h-8 w-8 text-primary" />
              ការព្យាករណ៍ការប្រើប្រាស់អគ្គិសនី
            </CardTitle>
            <CardDescription>
              Power Consumption Forecasting & Cost Analysis
            </CardDescription>
          </CardHeader>

          <div className="flex border-b mb-6">
            <button
              onClick={() => setActiveTab("prediction")}
              className={`px-4 py-2 font-medium text-sm ${
                activeTab === "prediction"
                  ? "border-b-2 border-primary text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <SparklesIcon className="h-4 w-4 inline mr-2" />
              Prediction
            </button>
            <button
              onClick={() => setActiveTab("cost")}
              className={`px-4 py-2 font-medium text-sm ${
                activeTab === "cost"
                  ? "border-b-2 border-primary text-primary"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <CurrencyDollarIcon className="h-4 w-4 inline mr-2" />
              Cost Analysis
            </button>
          </div>

          <CardContent className="space-y-6">
            {activeTab === "prediction" && (
              <>
                <div className="space-y-2">
                  <Label htmlFor="dayOfWeek">ថ្ងៃនៃសប្តាហ៍ (Day of Week)</Label>
                  <Select
                    value={dayOfWeek.toString()}
                    onValueChange={(value) => setDayOfWeek(parseInt(value))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select day" />
                    </SelectTrigger>
                    <SelectContent>
                      {daysOfWeek.map((day) => (
                        <SelectItem
                          key={day.value}
                          value={day.value.toString()}
                        >
                          {day.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="hour">ម៉ោង (Hour) - 0-23</Label>
                  <Input
                    id="hour"
                    type="number"
                    min="0"
                    max="23"
                    value={hour}
                    onChange={(e) => setHour(e.target.value)}
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="lag1h">Lag 1h (kW)</Label>
                    <Input
                      id="lag1h"
                      type="number"
                      step="0.1"
                      value={lag1h}
                      onChange={(e) => setLag1h(e.target.value)}
                      placeholder="2.5"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="lag24h">Lag 24h (kW)</Label>
                    <Input
                      id="lag24h"
                      type="number"
                      step="0.1"
                      value={lag24h}
                      onChange={(e) => setLag24h(e.target.value)}
                      placeholder="2.8"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="rollingMean24h">
                      Rolling Mean 24h (kW)
                    </Label>
                    <Input
                      id="rollingMean24h"
                      type="number"
                      step="0.1"
                      value={rollingMean24h}
                      onChange={(e) => setRollingMean24h(e.target.value)}
                      placeholder="2.3"
                    />
                  </div>
                </div>

                <Button
                  onClick={handlePredict}
                  disabled={loading}
                  className="w-full"
                  size="lg"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                      កំពុងព្យាករណ៍...
                    </>
                  ) : (
                    <>
                      <SparklesIcon className="h-4 w-4 mr-2" />
                      ព្យាករណ៍ (Predict)
                    </>
                  )}
                </Button>

                {prediction && (
                  <Card className="bg-green-50 border-green-200">
                    <CardHeader>
                      <CardTitle className="text-green-800">
                        លទ្ធផលព្យាករណ៍ (Prediction Result)
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-center mb-6">
                        <div className="text-4xl font-bold text-green-600">
                          {prediction.prediction.toFixed(2)} kW
                        </div>
                        <p className="text-green-700 mt-2">
                          ប្រើប្រាស់អគ្គិសនីរំពឹងទុក (Expected Power Usage)
                        </p>
                      </div>

                      <div className="bg-background rounded-lg p-4">
                        <h4 className="font-semibold text-foreground mb-3">
                          វិភាគម៉ូដែល (Model Analysis):
                        </h4>
                        <div className="space-y-2 text-sm text-muted-foreground">
                          <p>
                            <span className="font-medium">ថ្ងៃស្អែក:</span>{" "}
                            {
                              daysOfWeek.find(
                                (d) =>
                                  d.value === prediction.features_used.dayofweek
                              )?.label
                            }
                          </p>
                          <p>
                            <span className="font-medium">ម៉ោង:</span>{" "}
                            {prediction.features_used.hour}:00
                          </p>
                          <p>
                            <span className="font-medium">ម៉ោងមុន:</span>{" "}
                            {prediction.features_used.lag_1h} kW
                          </p>
                          <p>
                            <span className="font-medium">ម៉ោងដូចថ្ងៃមិញ:</span>{" "}
                            {prediction.features_used.lag_24h} kW
                          </p>
                          <p>
                            <span className="font-medium">មធ្យម 24 ម៉ោង:</span>{" "}
                            {prediction.features_used.rolling_mean_24h} kW
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </>
            )}

            {activeTab === "cost" && (
              <>
                <div className="space-y-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <CurrencyDollarIcon className="h-5 w-5" />
                        Cost Settings
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className="space-y-2">
                          <Label>Price per kWh ($)</Label>
                          <Input
                            type="number"
                            step="0.01"
                            value={costSettings.price_per_kwh}
                            onChange={(e) =>
                              setCostSettings((prev) => ({
                                ...prev,
                                price_per_kwh: parseFloat(e.target.value) || 0,
                              }))
                            }
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Peak Hours Start</Label>
                          <Input
                            type="number"
                            min="0"
                            max="23"
                            value={costSettings.peak_hours[0]}
                            onChange={(e) =>
                              setCostSettings((prev) => ({
                                ...prev,
                                peak_hours: [
                                  parseInt(e.target.value) || 0,
                                  prev.peak_hours[1],
                                ],
                              }))
                            }
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Peak Hours End</Label>
                          <Input
                            type="number"
                            min="0"
                            max="23"
                            value={costSettings.peak_hours[1]}
                            onChange={(e) =>
                              setCostSettings((prev) => ({
                                ...prev,
                                peak_hours: [
                                  prev.peak_hours[0],
                                  parseInt(e.target.value) || 0,
                                ],
                              }))
                            }
                          />
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>Peak Multiplier</Label>
                        <Input
                          type="number"
                          step="0.1"
                          value={costSettings.peak_multiplier}
                          onChange={(e) =>
                            setCostSettings((prev) => ({
                              ...prev,
                              peak_multiplier: parseFloat(e.target.value) || 1,
                            }))
                          }
                        />
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Usage Thresholds</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <Label>Hourly (kW)</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={costSettings.thresholds.hourly_kW}
                            onChange={(e) =>
                              setCostSettings((prev) => ({
                                ...prev,
                                thresholds: {
                                  ...prev.thresholds,
                                  hourly_kW: parseFloat(e.target.value) || 0,
                                },
                              }))
                            }
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Daily (kWh)</Label>
                          <Input
                            type="number"
                            step="1"
                            value={costSettings.thresholds.daily_kWh}
                            onChange={(e) =>
                              setCostSettings((prev) => ({
                                ...prev,
                                thresholds: {
                                  ...prev.thresholds,
                                  daily_kWh: parseFloat(e.target.value) || 0,
                                },
                              }))
                            }
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Weekly (kWh)</Label>
                          <Input
                            type="number"
                            step="1"
                            value={costSettings.thresholds.weekly_kWh}
                            onChange={(e) =>
                              setCostSettings((prev) => ({
                                ...prev,
                                thresholds: {
                                  ...prev.thresholds,
                                  weekly_kWh: parseFloat(e.target.value) || 0,
                                },
                              }))
                            }
                          />
                        </div>
                        <div className="space-y-2">
                          <Label>Monthly (kWh)</Label>
                          <Input
                            type="number"
                            step="1"
                            value={costSettings.thresholds.monthly_kWh}
                            onChange={(e) =>
                              setCostSettings((prev) => ({
                                ...prev,
                                thresholds: {
                                  ...prev.thresholds,
                                  monthly_kWh: parseFloat(e.target.value) || 0,
                                },
                              }))
                            }
                          />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle>Prediction Settings</CardTitle>
                      <CardDescription>
                        Configure how far ahead to predict costs
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <Label>Days to Predict Ahead</Label>
                        <Input
                          type="number"
                          value={daysAhead as number}
                          onChange={(e) => handleChangeDaysAhead(e)}
                        />
                        <p className="text-sm text-muted-foreground">
                          The system will use your trained model to predict
                          future power consumption and calculate costs.
                        </p>
                      </div>
                    </CardContent>
                  </Card>

                  <Button
                    onClick={handleCostAnalysis}
                    disabled={loading}
                    className="w-full"
                    size="lg"
                  >
                    {loading ? (
                      <>
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                        Predicting Costs...
                      </>
                    ) : (
                      <>
                        <ChartBarIcon className="h-4 w-4 mr-2" />
                        Predict Future Costs
                      </>
                    )}
                  </Button>

                  {costAnalysis && (
                    <Card className="bg-blue-50 border-blue-200">
                      <CardHeader>
                        <CardTitle className="text-blue-800">
                          Cost Prediction Results (
                          {costAnalysis.projections.days_predicted} days)
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                          <div className="text-center p-4 bg-white rounded-lg">
                            <div className="text-2xl font-bold text-blue-600">
                              $
                              {costAnalysis.projections.daily_avg_cost.toFixed(
                                2
                              )}
                            </div>
                            <p className="text-sm text-blue-700">
                              Avg Daily Cost
                            </p>
                          </div>
                          <div className="text-center p-4 bg-white rounded-lg">
                            <div className="text-2xl font-bold text-green-600">
                              $
                              {costAnalysis.projections.monthly_total_cost.toFixed(
                                2
                              )}
                            </div>
                            <p className="text-sm text-green-700">
                              Monthly Cost
                            </p>
                          </div>
                          <div className="text-center p-4 bg-white rounded-lg">
                            <div className="text-2xl font-bold text-purple-600">
                              $
                              {costAnalysis.projections.yearly_total_cost.toFixed(
                                2
                              )}
                            </div>
                            <p className="text-sm text-purple-700">
                              Yearly Cost
                            </p>
                          </div>
                          <div className="text-center p-4 bg-white rounded-lg">
                            <div className="text-2xl font-bold text-orange-600">
                              {costAnalysis.projections.monthly_avg_kwh.toFixed(
                                1
                              )}{" "}
                              kWh
                            </div>
                            <p className="text-sm text-orange-700">
                              Monthly Usage
                            </p>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4 mb-6">
                          <div className="text-center p-4 bg-yellow-50 rounded-lg">
                            <div className="text-xl font-bold text-yellow-600">
                              {costAnalysis.projections.peak_hours_daily.toFixed(
                                1
                              )}{" "}
                              kWh
                            </div>
                            <p className="text-sm text-yellow-700">
                              Peak Hours Daily
                            </p>
                          </div>
                          <div className="text-center p-4 bg-green-50 rounded-lg">
                            <div className="text-xl font-bold text-green-600">
                              {costAnalysis.projections.off_peak_hours_daily.toFixed(
                                1
                              )}{" "}
                              kWh
                            </div>
                            <p className="text-sm text-green-700">
                              Off-Peak Hours Daily
                            </p>
                          </div>
                        </div>

                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                          <div className="text-center">
                            <div className="text-lg font-bold text-blue-600">
                              $
                              {costAnalysis.summary.total_predicted_cost.toFixed(
                                2
                              )}
                            </div>
                            <p className="text-sm text-blue-700">
                              Total Predicted Cost
                            </p>
                          </div>
                          <div className="text-center">
                            <div className="text-lg font-bold text-blue-600">
                              {costAnalysis.summary.total_predicted_kwh.toFixed(
                                1
                              )}
                            </div>
                            <p className="text-sm text-blue-700">
                              Total Predicted kWh
                            </p>
                          </div>
                          <div className="text-center">
                            <div className="text-lg font-bold text-blue-600">
                              ${costAnalysis.summary.avg_daily_cost.toFixed(2)}
                            </div>
                            <p className="text-sm text-blue-700">
                              Avg Daily Cost
                            </p>
                          </div>
                          <div className="text-center">
                            <div className="text-lg font-bold text-red-600">
                              {costAnalysis.summary.total_violations}
                            </div>
                            <p className="text-sm text-red-700">
                              Threshold Violations
                            </p>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
              </>
            )}

            {error && (
              <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-4 flex items-center gap-2">
                <ExclamationTriangleIcon className="h-5 w-5 text-destructive" />
                <p className="text-destructive">{error}</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
