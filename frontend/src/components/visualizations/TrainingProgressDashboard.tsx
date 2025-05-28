import React, { useEffect, useState } from 'react';
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  RadialBarChart,
  RadialBar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie,
} from 'recharts';
import { TrainingStatus, ActionRecorded, PrincipleDiscovered } from '../../api/types';

interface TrainingProgressDashboardProps {
  status: TrainingStatus;
  actions: ActionRecorded[];
  principlesDiscovered: PrincipleDiscovered[];
  behavioralEntropy: number;
  onExport?: () => void;
}

export const TrainingProgressDashboard: React.FC<TrainingProgressDashboardProps> = ({
  status,
  actions,
  principlesDiscovered,
  behavioralEntropy,
  onExport,
}) => {
  const [entropyHistory, setEntropyHistory] = useState<{ time: string; value: number }[]>([]);
  const [actionFeed, setActionFeed] = useState<ActionRecorded[]>([]);

  useEffect(() => {
    // Update entropy history
    const now = new Date().toLocaleTimeString();
    setEntropyHistory(prev => [...prev.slice(-20), { time: now, value: behavioralEntropy }]);
  }, [behavioralEntropy]);

  useEffect(() => {
    // Update action feed with latest actions
    setActionFeed(actions.slice(-10).reverse());
  }, [actions]);

  // Prepare data for visualizations
  const progressData = {
    scenarios: {
      completed: status.scenarios_completed,
      total: status.scenarios_total,
      percentage: (status.scenarios_completed / status.scenarios_total) * 100,
    },
  };

  const principleStrengthData = principlesDiscovered.map(p => ({
    name: p.principle.name,
    strength: p.principle.strength * 100,
    consistency: p.principle.consistency * 100,
  }));

  const contextDistribution = actions.reduce((acc, action) => {
    acc[action.context] = (acc[action.context] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const contextData = Object.entries(contextDistribution).map(([context, count]) => ({
    name: context,
    value: count,
  }));

  const entropyGaugeData = [
    {
      name: 'Entropy',
      value: behavioralEntropy * 100,
      fill: behavioralEntropy > 0.7 ? '#ef4444' : behavioralEntropy > 0.4 ? '#f59e0b' : '#10b981',
    },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 p-4">
      {/* Progress Overview */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Training Progress</h3>
        <div className="space-y-4">
          <div>
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Scenarios</span>
              <span>{status.scenarios_completed} / {status.scenarios_total}</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progressData.scenarios.percentage}%` }}
              />
            </div>
          </div>
          
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-600">
              {progressData.scenarios.percentage.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-500">Complete</div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="bg-gray-50 p-2 rounded">
              <div className="text-gray-500">Status</div>
              <div className="font-semibold capitalize">{status.status}</div>
            </div>
            <div className="bg-gray-50 p-2 rounded">
              <div className="text-gray-500">Duration</div>
              <div className="font-semibold">
                {Math.floor((Date.now() - new Date(status.started_at).getTime()) / 60000)} min
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Behavioral Entropy Gauge */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Behavioral Entropy</h3>
        <ResponsiveContainer width="100%" height={200}>
          <RadialBarChart cx="50%" cy="50%" innerRadius="60%" outerRadius="90%" data={entropyGaugeData}>
            <RadialBar dataKey="value" cornerRadius={10} fill="#8884d8" />
            <text x="50%" y="50%" textAnchor="middle" dominantBaseline="middle" className="text-2xl font-bold">
              {(behavioralEntropy * 100).toFixed(1)}%
            </text>
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="text-center mt-2">
          <span className={`text-sm ${
            behavioralEntropy > 0.7 ? 'text-red-500' : 
            behavioralEntropy > 0.4 ? 'text-yellow-500' : 
            'text-green-500'
          }`}>
            {behavioralEntropy > 0.7 ? 'High Variability' : 
             behavioralEntropy > 0.4 ? 'Moderate Consistency' : 
             'High Consistency'}
          </span>
        </div>
      </div>

      {/* Principles Discovered Counter */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Principles Discovered</h3>
        <div className="text-center">
          <div className="text-5xl font-bold text-purple-600">
            {principlesDiscovered.length}
          </div>
          <div className="text-sm text-gray-500 mt-2">Unique Principles</div>
        </div>
        <div className="mt-4 space-y-2">
          {principlesDiscovered.slice(-3).map((p, i) => (
            <div key={i} className="text-xs bg-purple-50 p-2 rounded">
              <div className="font-semibold">{p.principle.name}</div>
              <div className="text-gray-500">
                Strength: {(p.principle.strength * 100).toFixed(0)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Entropy History Chart */}
      <div className="bg-white p-4 rounded-lg shadow lg:col-span-2">
        <h3 className="text-lg font-semibold mb-4">Entropy Over Time</h3>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={entropyHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" />
            <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#8b5cf6"
              fill="#8b5cf6"
              fillOpacity={0.3}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Context Distribution */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Decision Contexts</h3>
        <ResponsiveContainer width="100%" height={200}>
          <PieChart>
            <Pie
              data={contextData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {contextData.map((_entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Action Feed */}
      <div className="bg-white p-4 rounded-lg shadow lg:col-span-3">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold">Recent Actions</h3>
          <button
            onClick={onExport}
            className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
          >
            Export Dashboard
          </button>
        </div>
        <div className="space-y-2 max-h-60 overflow-y-auto">
          {actionFeed.map((action, i) => (
            <div key={i} className="flex items-start space-x-3 text-sm border-b pb-2">
              <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-1.5"></div>
              <div className="flex-1">
                <div className="font-medium">{action.action}</div>
                <div className="text-gray-500 text-xs">
                  Context: {action.context} | Decision: {action.decision}
                </div>
              </div>
              <div className="text-gray-400 text-xs">
                {new Date(action.timestamp).toLocaleTimeString()}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Principle Strength Comparison */}
      {principleStrengthData.length > 0 && (
        <div className="bg-white p-4 rounded-lg shadow lg:col-span-3">
          <h3 className="text-lg font-semibold mb-4">Principle Strength & Consistency</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={principleStrengthData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
              <YAxis tickFormatter={(v) => `${v}%`} />
              <Tooltip formatter={(value: number) => `${value.toFixed(1)}%`} />
              <Legend />
              <Bar dataKey="strength" fill="#8b5cf6" name="Strength" />
              <Bar dataKey="consistency" fill="#10b981" name="Consistency" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// Color palette for pie chart
const COLORS = ['#8b5cf6', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#06b6d4', '#84cc16'];
