import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { PrincipleReport } from '../../api/types';

interface AgentPrinciples {
  agentId: string;
  agentName: string;
  principles: PrincipleReport[];
}

interface AgentComparisonMatrixProps {
  agents: AgentPrinciples[];
  metric?: 'strength' | 'consistency' | 'volatility';
  onCellClick?: (agentId: string, principleName: string) => void;
}

export const AgentComparisonMatrix: React.FC<AgentComparisonMatrixProps> = ({
  agents,
  metric = 'strength',
  onCellClick,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedCell, setSelectedCell] = useState<{ agent: string; principle: string } | null>(null);
  const [sortBy, setSortBy] = useState<'agent' | 'principle' | 'value'>('agent');

  useEffect(() => {
    if (!svgRef.current || !agents.length) return;

    const margin = { top: 150, right: 100, bottom: 50, left: 150 };
    const cellSize = 40;
    
    // Extract all unique principles
    const allPrinciples = Array.from(new Set(
      agents.flatMap(a => a.principles.map(p => p.name))
    ));

    // Calculate volatility if needed
    const calculateVolatility = (values: number[]) => {
      if (values.length < 2) return 0;
      const mean = values.reduce((a, b) => a + b, 0) / values.length;
      const variance = values.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / values.length;
      return Math.sqrt(variance);
    };

    // Prepare data matrix
    const matrix = agents.map(agent => {
      const principleMap = new Map(
        agent.principles.map(p => [p.name, p])
      );
      
      return {
        agentId: agent.agentId,
        agentName: agent.agentName,
        values: allPrinciples.map(principleName => {
          const principle = principleMap.get(principleName);
          if (!principle) return null;
          
          switch (metric) {
            case 'strength':
              return principle.strength;
            case 'consistency':
              return principle.consistency;
            case 'volatility':
              // For volatility, we'd need historical data
              // This is a placeholder calculation
              return calculateVolatility([principle.strength, principle.consistency]);
            default:
              return principle.strength;
          }
        }),
      };
    });

    // Sort based on current sorting preference
    if (sortBy === 'value') {
      // Sort by average value across all agents
      const principleAverages = allPrinciples.map((p, i) => ({
        index: i,
        average: d3.mean(matrix, d => d.values[i] || 0) || 0,
      }));
      principleAverages.sort((a, b) => b.average - a.average);
      
      // Reorder principles
      const newOrder = principleAverages.map(p => p.index);
      allPrinciples.sort((a, b) => {
        const aIndex = allPrinciples.indexOf(a);
        const bIndex = allPrinciples.indexOf(b);
        return newOrder.indexOf(aIndex) - newOrder.indexOf(bIndex);
      });
    }

    const width = margin.left + margin.right + cellSize * allPrinciples.length;
    const height = margin.top + margin.bottom + cellSize * agents.length;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Color scale
    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 1]);

    // Create cells
    const cells = g.selectAll('.cell')
      .data(matrix.flatMap((agent, i) => 
        agent.values.map((value, j) => ({
          agentId: agent.agentId,
          agentName: agent.agentName,
          principle: allPrinciples[j],
          value,
          x: j * cellSize,
          y: i * cellSize,
        }))
      ))
      .enter().append('g')
      .attr('class', 'cell')
      .attr('transform', d => `translate(${d.x}, ${d.y})`);

    // Add rectangles
    cells.append('rect')
      .attr('width', cellSize - 2)
      .attr('height', cellSize - 2)
      .attr('fill', d => d.value !== null ? colorScale(d.value) : '#f0f0f0')
      .attr('stroke', d => 
        selectedCell && selectedCell.agent === d.agentId && selectedCell.principle === d.principle
          ? '#000' : '#fff'
      )
      .attr('stroke-width', d => 
        selectedCell && selectedCell.agent === d.agentId && selectedCell.principle === d.principle
          ? 2 : 1
      )
      .style('cursor', 'pointer')
      .on('click', (event, d) => {
        if (d.value !== null) {
          setSelectedCell({ agent: d.agentId, principle: d.principle });
          if (onCellClick) {
            onCellClick(d.agentId, d.principle);
          }
        }
      })
      .on('mouseenter', function(event, d) {
        if (d.value !== null) {
          d3.select(this).attr('stroke', '#000').attr('stroke-width', 2);
          
          // Highlight row and column
          g.selectAll('.cell rect')
            .attr('opacity', (cellData: any) => {
              return cellData.agentId === d.agentId || cellData.principle === d.principle
                ? 1 : 0.3;
            });
        }
      })
      .on('mouseleave', function(event, d) {
        d3.select(this)
          .attr('stroke', selectedCell && selectedCell.agent === d.agentId && selectedCell.principle === d.principle
            ? '#000' : '#fff')
          .attr('stroke-width', selectedCell && selectedCell.agent === d.agentId && selectedCell.principle === d.principle
            ? 2 : 1);
        
        g.selectAll('.cell rect').attr('opacity', 1);
      });

    // Add text values
    cells.append('text')
      .attr('x', cellSize / 2)
      .attr('y', cellSize / 2)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .style('font-size', '10px')
      .style('fill', d => d.value !== null && d.value > 0.5 ? '#fff' : '#000')
      .text(d => d.value !== null ? (d.value * 100).toFixed(0) : '');

    // Add agent labels
    g.selectAll('.agent-label')
      .data(matrix)
      .enter().append('text')
      .attr('class', 'agent-label')
      .attr('x', -10)
      .attr('y', (d, i) => i * cellSize + cellSize / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .style('font-size', '12px')
      .style('cursor', 'pointer')
      .text(d => d.agentName)
      .on('click', () => setSortBy('agent'));

    // Add principle labels
    g.selectAll('.principle-label')
      .data(allPrinciples)
      .enter().append('text')
      .attr('class', 'principle-label')
      .attr('transform', (d, i) => `translate(${i * cellSize + cellSize / 2}, -10) rotate(-45)`)
      .attr('text-anchor', 'start')
      .style('font-size', '12px')
      .style('cursor', 'pointer')
      .text(d => d)
      .on('click', () => setSortBy('principle'));

    // Add color legend
    const legendWidth = 200;
    const legendHeight = 20;
    
    const legend = svg.append('g')
      .attr('transform', `translate(${width - legendWidth - 20}, 20)`);

    const legendScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => `${(d as number * 100).toFixed(0)}%`);

    // Create gradient for legend
    const gradientId = `gradient-${metric}`;
    const gradient = svg.append('defs')
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '100%')
      .attr('y1', '0%')
      .attr('y2', '0%');

    const steps = 20;
    for (let i = 0; i <= steps; i++) {
      gradient.append('stop')
        .attr('offset', `${(i / steps) * 100}%`)
        .attr('stop-color', colorScale(i / steps));
    }

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', `url(#${gradientId})`);

    legend.append('g')
      .attr('transform', `translate(0, ${legendHeight})`)
      .call(legendAxis);

    legend.append('text')
      .attr('x', legendWidth / 2)
      .attr('y', -5)
      .attr('text-anchor', 'middle')
      .style('font-size', '12px')
      .style('font-weight', 'bold')
      .text(metric.charAt(0).toUpperCase() + metric.slice(1));

  }, [agents, metric, selectedCell, sortBy, onCellClick]);

  return (
    <div className="relative">
      <div className="absolute top-0 right-0 flex gap-2">
        <select
          value={metric}
          onChange={(e) => setSortBy('agent')}
          className="px-3 py-1 border rounded text-sm"
        >
          <option value="strength">Strength</option>
          <option value="consistency">Consistency</option>
          <option value="volatility">Volatility</option>
        </select>
        <button
          onClick={() => setSortBy(sortBy === 'value' ? 'agent' : 'value')}
          className="px-3 py-1 bg-gray-200 rounded hover:bg-gray-300 text-sm"
        >
          Sort by {sortBy === 'value' ? 'Agent' : 'Value'}
        </button>
      </div>
      <div className="overflow-auto mt-10">
        <svg ref={svgRef}></svg>
      </div>
      {selectedCell && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-600">
            Click on any cell to drill down into specific agent-principle details.
          </p>
        </div>
      )}
    </div>
  );
};
