import React, { useEffect, useRef, useState } from 'react';
import { Network, DataSet, Node, Edge } from 'vis-network/standalone';
import 'vis-network/styles/vis-network.css';

interface BehavioralNode {
  id: string;
  action: string;
  context: string;
  decision: string;
  frequency: number;
}

interface BehavioralEdge {
  from: string;
  to: string;
  weight: number;
  context?: string;
}

interface BehavioralPatternNetworkProps {
  nodes: BehavioralNode[];
  edges: BehavioralEdge[];
  height?: string;
  onNodeClick?: (nodeId: string) => void;
}

export const BehavioralPatternNetwork: React.FC<BehavioralPatternNetworkProps> = ({
  nodes,
  edges,
  height = '600px',
  onNodeClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [network, setNetwork] = useState<Network | null>(null);

  useEffect(() => {
    if (!containerRef.current || !nodes.length) return;

    // Prepare nodes with visual properties
    const visNodes = new DataSet<Node>(
      nodes.map(node => ({
        id: node.id,
        label: node.action,
        title: `${node.action}\nContext: ${node.context}\nDecision: ${node.decision}\nFrequency: ${node.frequency}`,
        value: node.frequency, // Size based on frequency
        color: {
          background: getContextColor(node.context),
          border: '#2B7CE9',
          highlight: {
            background: getContextColor(node.context, true),
            border: '#1A5490',
          },
        },
        font: {
          color: '#343434',
          size: 12,
        },
      }))
    );

    // Prepare edges with visual properties
    const visEdges = new DataSet<Edge>(
      edges.map(edge => ({
        from: edge.from,
        to: edge.to,
        value: edge.weight,
        title: `Weight: ${edge.weight.toFixed(2)}${edge.context ? `\nContext: ${edge.context}` : ''}`,
        arrows: {
          to: {
            enabled: true,
            scaleFactor: 0.5,
          },
        },
        color: {
          color: '#848484',
          highlight: '#2B7CE9',
          opacity: Math.min(0.3 + edge.weight * 0.7, 1),
        },
        smooth: {
          enabled: true,
          type: 'curvedCW',
          roundness: 0.2,
        },
      }))
    );

    // Network options
    const options = {
      nodes: {
        shape: 'dot',
        scaling: {
          min: 10,
          max: 30,
          label: {
            min: 8,
            max: 14,
            drawThreshold: 5,
            maxVisible: 20,
          },
        },
        borderWidth: 2,
        shadow: true,
      },
      edges: {
        width: 2,
        shadow: true,
        smooth: {
          enabled: true,
          type: 'dynamic',
          roundness: 0.5,
        },
      },
      physics: {
        forceAtlas2Based: {
          gravitationalConstant: -50,
          centralGravity: 0.01,
          springLength: 100,
          springConstant: 0.08,
          damping: 0.4,
          avoidOverlap: 0.5,
        },
        maxVelocity: 50,
        solver: 'forceAtlas2Based',
        timestep: 0.35,
        stabilization: {
          enabled: true,
          iterations: 1000,
          updateInterval: 100,
          onlyDynamicEdges: false,
          fit: true,
        },
      },
      interaction: {
        hover: true,
        tooltipDelay: 200,
        navigationButtons: true,
        keyboard: true,
      },
      layout: {
        improvedLayout: true,
        clusterThreshold: 150,
      },
    };

    // Create network
    const net = new Network(
      containerRef.current,
      { nodes: visNodes, edges: visEdges },
      options
    );

    // Event handlers
    net.on('click', (params) => {
      if (params.nodes.length > 0) {
        const nodeId = params.nodes[0];
        setSelectedNode(nodeId);
        if (onNodeClick) {
          onNodeClick(nodeId);
        }
      } else {
        setSelectedNode(null);
      }
    });

    net.on('doubleClick', (params) => {
      if (params.nodes.length > 0) {
        net.focus(params.nodes[0], {
          scale: 1.5,
          animation: {
            duration: 1000,
            easingFunction: 'easeInOutQuad',
          },
        });
      }
    });

    setNetwork(net);

    // Cleanup
    return () => {
      if (net) {
        net.destroy();
      }
    };
  }, [nodes, edges, onNodeClick]);

  // Helper function to determine node color based on context
  const getContextColor = (context: string, highlight = false): string => {
    const colors: { [key: string]: { normal: string; highlight: string } } = {
      'self-preservation': { normal: '#FF6B6B', highlight: '#FF5252' },
      'harm-prevention': { normal: '#4ECDC4', highlight: '#45B7AA' },
      'fairness': { normal: '#FFD93D', highlight: '#FFCE3D' },
      'loyalty': { normal: '#95E1D3', highlight: '#81D4C4' },
      'authority': { normal: '#C7CEEA', highlight: '#B5BFEA' },
      'efficiency': { normal: '#FFA502', highlight: '#FF9502' },
      'care': { normal: '#FF6348', highlight: '#FF4834' },
      'integrity': { normal: '#7BED9F', highlight: '#5BE881' },
    };

    const colorSet = colors[context.toLowerCase()] || { normal: '#97A2B0', highlight: '#7C8694' };
    return highlight ? colorSet.highlight : colorSet.normal;
  };

  // Cluster detection
  const detectClusters = () => {
    if (!network) return;
    
    // Implement community detection algorithm
    network.cluster({
      joinCondition: (nodeOptions: any) => {
        return nodeOptions.value > 5; // Cluster high-frequency nodes
      },
      clusterNodeProperties: {
        borderWidth: 3,
        shape: 'database',
        color: '#FF9F40',
        label: 'Behavioral Cluster',
      },
    });
  };

  return (
    <div className="relative">
      <div className="absolute top-2 right-2 z-10 space-x-2">
        <button
          onClick={() => network?.fit()}
          className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
        >
          Fit View
        </button>
        <button
          onClick={detectClusters}
          className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600 text-sm"
        >
          Detect Clusters
        </button>
      </div>
      
      <div 
        ref={containerRef} 
        style={{ height }} 
        className="border border-gray-300 rounded-lg"
      />
      
      {selectedNode && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h4 className="font-semibold text-gray-700">Selected Node: {selectedNode}</h4>
          <p className="text-sm text-gray-600 mt-1">
            Double-click to focus, or use the navigation controls to explore the network.
          </p>
        </div>
      )}

      <div className="mt-4 grid grid-cols-4 gap-2 text-xs">
        <div className="flex items-center">
          <div className="w-4 h-4 rounded-full bg-red-400 mr-2"></div>
          <span>Self-preservation</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 rounded-full bg-teal-400 mr-2"></div>
          <span>Harm-prevention</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 rounded-full bg-yellow-400 mr-2"></div>
          <span>Fairness</span>
        </div>
        <div className="flex items-center">
          <div className="w-4 h-4 rounded-full bg-green-300 mr-2"></div>
          <span>Loyalty</span>
        </div>
      </div>
    </div>
  );
};
