import React, { useState, useEffect, ChangeEvent } from 'react';
import { apiClient } from '../api/client';

interface PluginMetadata {
  name: string;
  version: string;
  author: string;
  description: string;
  plugin_type: 'inference' | 'scenario' | 'analysis';
  dependencies: string[];
  config_schema?: Record<string, any>;
  tags: string[];
  created_at: string;
}

interface PluginConfig {
  [key: string]: any;
}

interface Plugin {
  metadata: PluginMetadata;
  enabled: boolean;
  config: PluginConfig;
  instance_loaded: boolean;
}

interface PluginsByType {
  inference: Plugin[];
  scenario: Plugin[];
  analysis: Plugin[];
}

const PluginMarketplace: React.FC = () => {
  const [plugins, setPlugins] = useState<PluginsByType>({
    inference: [],
    scenario: [],
    analysis: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedPlugin, setSelectedPlugin] = useState<Plugin | null>(null);
  const [configDialogOpen, setConfigDialogOpen] = useState(false);
  const [tempConfig, setTempConfig] = useState<PluginConfig>({});
  const [loadingStates, setLoadingStates] = useState<Record<string, boolean>>({});
  const [activeTab, setActiveTab] = useState<'inference' | 'scenario' | 'analysis'>('inference');

  useEffect(() => {
    fetchPlugins();
  }, []);

  const fetchPlugins = async () => {
    try {
      setLoading(true);
      setError(null);

      // Get list of all plugins
      const response = await apiClient.get('/api/plugins/list');
      const pluginList = response.data;

      // Get detailed metadata for each plugin
      const pluginsByType: PluginsByType = {
        inference: [],
        scenario: [],
        analysis: []
      };

      for (const [type, names] of Object.entries(pluginList)) {
        for (const name of names as string[]) {
          try {
            const metadataResponse = await apiClient.get(
              `/api/plugins/metadata/${type}/${name}`
            );
            
            const plugin: Plugin = {
              metadata: metadataResponse.data,
              enabled: false,
              config: {},
              instance_loaded: false
            };

            pluginsByType[type as keyof PluginsByType].push(plugin);
          } catch (err) {
            console.error(`Failed to fetch metadata for ${name}:`, err);
          }
        }
      }

      setPlugins(pluginsByType);
    } catch (err) {
      setError('Failed to fetch plugins');
      console.error('Error fetching plugins:', err);
    } finally {
      setLoading(false);
    }
  };

  const togglePlugin = async (plugin: Plugin) => {
    const loadingKey = `${plugin.metadata.plugin_type}-${plugin.metadata.name}`;
    setLoadingStates(prev => ({ ...prev, [loadingKey]: true }));

    try {
      if (!plugin.enabled) {
        // Load plugin
        await apiClient.post('/api/plugins/load', {
          plugin_type: plugin.metadata.plugin_type,
          plugin_name: plugin.metadata.name,
          config: plugin.config
        });

        // Update plugin state
        setPlugins(prev => ({
          ...prev,
          [plugin.metadata.plugin_type]: prev[plugin.metadata.plugin_type].map(p =>
            p.metadata.name === plugin.metadata.name
              ? { ...p, enabled: true, instance_loaded: true }
              : p
          )
        }));
      } else {
        // For now, we can't unload plugins via API, so just update UI
        setPlugins(prev => ({
          ...prev,
          [plugin.metadata.plugin_type]: prev[plugin.metadata.plugin_type].map(p =>
            p.metadata.name === plugin.metadata.name
              ? { ...p, enabled: false }
              : p
          )
        }));
      }
    } catch (err) {
      console.error('Error toggling plugin:', err);
      setError(`Failed to ${plugin.enabled ? 'disable' : 'enable'} plugin`);
    } finally {
      setLoadingStates(prev => ({ ...prev, [loadingKey]: false }));
    }
  };

  const openConfigDialog = (plugin: Plugin) => {
    setSelectedPlugin(plugin);
    setTempConfig(plugin.config);
    setConfigDialogOpen(true);
  };

  const savePluginConfig = async () => {
    if (!selectedPlugin) return;

    try {
      // If plugin is enabled, reload with new config
      if (selectedPlugin.enabled) {
        await apiClient.post('/api/plugins/load', {
          plugin_type: selectedPlugin.metadata.plugin_type,
          plugin_name: selectedPlugin.metadata.name,
          config: tempConfig
        });
      }

      // Update local state
      setPlugins(prev => ({
        ...prev,
        [selectedPlugin.metadata.plugin_type]: prev[selectedPlugin.metadata.plugin_type].map(p =>
          p.metadata.name === selectedPlugin.metadata.name
            ? { ...p, config: tempConfig }
            : p
        )
      }));

      setConfigDialogOpen(false);
    } catch (err) {
      console.error('Error saving plugin config:', err);
      setError('Failed to save plugin configuration');
    }
  };

  const getPluginIcon = (type: string) => {
    switch (type) {
      case 'inference':
        return '🧠';
      case 'scenario':
        return '💻';
      case 'analysis':
        return '📊';
      default:
        return '📦';
    }
  };

  const renderConfigField = (key: string, schema: any, value: any) => {
    const fieldType = schema?.type || 'string';
    const isRequired = schema?.required || false;

    switch (fieldType) {
      case 'number':
      case 'float':
        return (
          <div key={key} style={{ marginBottom: '16px' }}>
            <label htmlFor={key} style={{ display: 'block', marginBottom: '4px', fontWeight: 500 }}>
              {key} {isRequired && <span style={{ color: 'red' }}>*</span>}
            </label>
            <input
              id={key}
              type="number"
              value={value || ''}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setTempConfig({
                ...tempConfig,
                [key]: parseFloat(e.target.value)
              })}
              placeholder={schema?.default?.toString() || ''}
              min={schema?.min}
              max={schema?.max}
              step={fieldType === 'float' ? 0.1 : 1}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            />
            {schema?.description && (
              <p style={{ fontSize: '14px', color: '#666', marginTop: '4px' }}>{schema.description}</p>
            )}
          </div>
        );

      case 'boolean':
        return (
          <div key={key} style={{ marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              id={key}
              type="checkbox"
              checked={value || false}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setTempConfig({
                ...tempConfig,
                [key]: e.target.checked
              })}
              style={{ width: '16px', height: '16px' }}
            />
            <label htmlFor={key} style={{ fontWeight: 500 }}>
              {key} {isRequired && <span style={{ color: 'red' }}>*</span>}
            </label>
          </div>
        );

      case 'array':
        return (
          <div key={key} style={{ marginBottom: '16px' }}>
            <label htmlFor={key} style={{ display: 'block', marginBottom: '4px', fontWeight: 500 }}>
              {key} {isRequired && <span style={{ color: 'red' }}>*</span>}
            </label>
            <textarea
              id={key}
              value={JSON.stringify(value || [], null, 2)}
              onChange={(e: ChangeEvent<HTMLTextAreaElement>) => {
                try {
                  const parsed = JSON.parse(e.target.value);
                  setTempConfig({ ...tempConfig, [key]: parsed });
                } catch {}
              }}
              placeholder="[]"
              rows={3}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontFamily: 'monospace'
              }}
            />
            {schema?.description && (
              <p style={{ fontSize: '14px', color: '#666', marginTop: '4px' }}>{schema.description}</p>
            )}
          </div>
        );

      default:
        return (
          <div key={key} style={{ marginBottom: '16px' }}>
            <label htmlFor={key} style={{ display: 'block', marginBottom: '4px', fontWeight: 500 }}>
              {key} {isRequired && <span style={{ color: 'red' }}>*</span>}
            </label>
            <input
              id={key}
              type="text"
              value={value || ''}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setTempConfig({
                ...tempConfig,
                [key]: e.target.value
              })}
              placeholder={schema?.default || ''}
              style={{
                width: '100%',
                padding: '8px',
                border: '1px solid #ddd',
                borderRadius: '4px'
              }}
            />
            {schema?.description && (
              <p style={{ fontSize: '14px', color: '#666', marginTop: '4px' }}>{schema.description}</p>
            )}
          </div>
        );
    }
  };

  const renderPluginCard = (plugin: Plugin) => {
    const loadingKey = `${plugin.metadata.plugin_type}-${plugin.metadata.name}`;
    const isLoading = loadingStates[loadingKey] || false;

    return (
      <div
        key={plugin.metadata.name}
        style={{
          border: '1px solid #ddd',
          borderRadius: '8px',
          padding: '16px',
          backgroundColor: '#fff',
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '20px' }}>{getPluginIcon(plugin.metadata.plugin_type)}</span>
            <h3 style={{ margin: 0, fontSize: '18px' }}>{plugin.metadata.name}</h3>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{
              padding: '2px 8px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              fontSize: '12px'
            }}>
              {plugin.metadata.version}
            </span>
            <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
              <input
                type="checkbox"
                checked={plugin.enabled}
                onChange={() => togglePlugin(plugin)}
                disabled={isLoading}
                style={{ marginRight: '4px' }}
              />
              Enable
            </label>
          </div>
        </div>

        <p style={{ color: '#666', fontSize: '14px', marginBottom: '12px' }}>
          {plugin.metadata.description}
        </p>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '14px' }}>
          <span style={{ color: '#999' }}>by {plugin.metadata.author}</span>
          <div style={{ display: 'flex', gap: '4px' }}>
            {plugin.metadata.tags.map(tag => (
              <span
                key={tag}
                style={{
                  padding: '2px 6px',
                  backgroundColor: '#f0f0f0',
                  borderRadius: '4px',
                  fontSize: '12px'
                }}
              >
                {tag}
              </span>
            ))}
          </div>
        </div>

        {plugin.metadata.dependencies.length > 0 && (
          <div style={{ marginTop: '12px', fontSize: '14px' }}>
            <strong>Dependencies:</strong>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px', marginTop: '4px' }}>
              {plugin.metadata.dependencies.map(dep => (
                <span
                  key={dep}
                  style={{
                    padding: '2px 6px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    fontSize: '12px'
                  }}
                >
                  {dep}
                </span>
              ))}
            </div>
          </div>
        )}

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px' }}>
          <div style={{ display: 'flex', gap: '8px' }}>
            {plugin.enabled && (
              <span style={{
                padding: '4px 8px',
                backgroundColor: '#4ade80',
                color: 'white',
                borderRadius: '4px',
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}>
                ✓ Active
              </span>
            )}
            {plugin.instance_loaded && (
              <span style={{
                padding: '4px 8px',
                border: '1px solid #ddd',
                borderRadius: '4px',
                fontSize: '12px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}>
                ⬇ Loaded
              </span>
            )}
          </div>

          <button
            onClick={() => openConfigDialog(plugin)}
            disabled={!plugin.metadata.config_schema}
            style={{
              padding: '6px 12px',
              border: '1px solid #ddd',
              borderRadius: '4px',
              backgroundColor: plugin.metadata.config_schema ? '#fff' : '#f5f5f5',
              cursor: plugin.metadata.config_schema ? 'pointer' : 'not-allowed',
              fontSize: '14px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            ⚙️ Configure
          </button>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '256px' }}>
        <div style={{ fontSize: '32px', animation: 'spin 1s linear infinite' }}>⟳</div>
      </div>
    );
  }

  return (
    <div style={{ maxWidth: '1280px', margin: '0 auto', padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
        <h1 style={{ fontSize: '32px', fontWeight: 'bold', margin: 0 }}>Plugin Marketplace</h1>
        <button
          onClick={fetchPlugins}
          style={{
            padding: '8px 16px',
            border: '1px solid #ddd',
            borderRadius: '4px',
            backgroundColor: '#fff',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          ⟳ Refresh
        </button>
      </div>

      {error && (
        <div style={{
          padding: '12px',
          backgroundColor: '#fee',
          border: '1px solid #fcc',
          borderRadius: '4px',
          marginBottom: '24px',
          color: '#c00'
        }}>
          {error}
        </div>
      )}

      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', borderBottom: '2px solid #ddd', marginBottom: '16px' }}>
          {(['inference', 'scenario', 'analysis'] as const).map(type => (
            <button
              key={type}
              onClick={() => setActiveTab(type)}
              style={{
                padding: '12px 24px',
                border: 'none',
                borderBottom: activeTab === type ? '2px solid #3b82f6' : '2px solid transparent',
                backgroundColor: 'transparent',
                cursor: 'pointer',
                fontSize: '16px',
                color: activeTab === type ? '#3b82f6' : '#666',
                fontWeight: activeTab === type ? 'bold' : 'normal',
                display: 'flex',
                alignItems: 'center',
                gap: '8px'
              }}
            >
              {getPluginIcon(type)} {type.charAt(0).toUpperCase() + type.slice(1)} ({plugins[type].length})
            </button>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '16px' }}>
          {plugins[activeTab].map(plugin => renderPluginCard(plugin))}
        </div>

        {plugins[activeTab].length === 0 && (
          <div style={{ textAlign: 'center', padding: '48px', color: '#999' }}>
            No {activeTab} plugins available
          </div>
        )}
      </div>

      {configDialogOpen && selectedPlugin && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.5)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000
        }}>
          <div style={{
            backgroundColor: '#fff',
            borderRadius: '8px',
            padding: '24px',
            maxWidth: '600px',
            width: '90%',
            maxHeight: '80vh',
            overflowY: 'auto'
          }}>
            <h2 style={{ margin: '0 0 8px 0' }}>Configure {selectedPlugin.metadata.name}</h2>
            <p style={{ color: '#666', marginBottom: '24px' }}>
              Customize the plugin settings below. Changes will be applied when you save.
            </p>

            <div style={{ marginBottom: '24px' }}>
              <h3 style={{ marginBottom: '16px' }}>Plugin Information</h3>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', fontSize: '14px' }}>
                <div>
                  <span style={{ color: '#666' }}>Version:</span>
                  <span style={{ marginLeft: '8px' }}>{selectedPlugin.metadata.version}</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Author:</span>
                  <span style={{ marginLeft: '8px' }}>{selectedPlugin.metadata.author}</span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Type:</span>
                  <span style={{ marginLeft: '8px', textTransform: 'capitalize' }}>
                    {selectedPlugin.metadata.plugin_type}
                  </span>
                </div>
                <div>
                  <span style={{ color: '#666' }}>Created:</span>
                  <span style={{ marginLeft: '8px' }}>
                    {new Date(selectedPlugin.metadata.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>

            {selectedPlugin.metadata.config_schema && (
              <div style={{ marginBottom: '24px' }}>
                <h3 style={{ marginBottom: '16px' }}>Configuration</h3>
                <div>
                  {Object.entries(selectedPlugin.metadata.config_schema).map(([key, schema]) =>
                    renderConfigField(key, schema, tempConfig[key])
                  )}
                </div>
              </div>
            )}

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
              <button
                onClick={() => setConfigDialogOpen(false)}
                style={{
                  padding: '8px 16px',
                  border: '1px solid #ddd',
                  borderRadius: '4px',
                  backgroundColor: '#fff',
                  cursor: 'pointer'
                }}
              >
                Cancel
              </button>
              <button
                onClick={savePluginConfig}
                style={{
                  padding: '8px 16px',
                  border: 'none',
                  borderRadius: '4px',
                  backgroundColor: '#3b82f6',
                  color: '#fff',
                  cursor: 'pointer'
                }}
              >
                Save Configuration
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default PluginMarketplace;
