import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface PrincipleEmergenceData {
  timestamp: Date;
  principles: {
    name: string;
    strength: number;
    consistency: number;
  }[];
}

interface PrincipleEmergenceGraphProps {
  data: PrincipleEmergenceData[];
  selectedPrinciples?: string[];
  height?: number;
  width?: number;
}

export const PrincipleEmergenceGraph: React.FC<PrincipleEmergenceGraphProps> = ({
  data,
  selectedPrinciples,
  height = 400,
  width = 800,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredPrinciple, setHoveredPrinciple] = useState<string | null>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    content: {
      principle: string;
      strength: number;
      consistency: number;
      timestamp: Date;
    } | null;
  }>({ x: 0, y: 0, content: null });

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    const margin = { top: 20, right: 120, bottom: 50, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove();

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Extract unique principles
    const allPrinciples = Array.from(new Set(
      data.flatMap(d => d.principles.map(p => p.name))
    ));
    
    const principles = selectedPrinciples?.length 
      ? allPrinciples.filter(p => selectedPrinciples.includes(p))
      : allPrinciples;

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => d.timestamp) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([innerHeight, 0]);

    const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
      .domain(principles);

    // Line generator
    const line = d3.line<{ timestamp: Date; strength: number }>()
      .x(d => xScale(d.timestamp))
      .y(d => yScale(d.strength))
      .curve(d3.curveMonotoneX);

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(xScale).tickFormat((d) => d3.timeFormat('%H:%M')(d as Date)))
      .append('text')
      .attr('x', innerWidth / 2)
      .attr('y', 40)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('Time');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -35)
      .attr('x', -innerHeight / 2)
      .attr('fill', 'black')
      .style('text-anchor', 'middle')
      .text('Principle Strength');

    // Draw lines for each principle
    principles.forEach(principle => {
      const principleData = data.map(d => ({
        timestamp: d.timestamp,
        strength: d.principles.find(p => p.name === principle)?.strength || 0,
        consistency: d.principles.find(p => p.name === principle)?.consistency || 0,
      }));

      // Draw the line
      g.append('path')
        .datum(principleData)
        .attr('fill', 'none')
        .attr('stroke', colorScale(principle))
        .attr('stroke-width', hoveredPrinciple === principle ? 3 : 2)
        .attr('opacity', hoveredPrinciple && hoveredPrinciple !== principle ? 0.3 : 1)
        .attr('d', line)
        .attr('class', `line-${principle.replace(/\s+/g, '-')}`);

      // Add dots for interaction
      g.selectAll(`.dot-${principle.replace(/\s+/g, '-')}`)
        .data(principleData)
        .enter().append('circle')
        .attr('class', `dot-${principle.replace(/\s+/g, '-')}`)
        .attr('cx', d => xScale(d.timestamp))
        .attr('cy', d => yScale(d.strength))
        .attr('r', 3)
        .attr('fill', colorScale(principle))
        .attr('opacity', hoveredPrinciple && hoveredPrinciple !== principle ? 0.3 : 1)
        .on('mouseenter', (event, d) => {
          setTooltip({
            x: event.pageX,
            y: event.pageY,
            content: {
              principle,
              strength: d.strength,
              consistency: d.consistency,
              timestamp: d.timestamp,
            },
          });
        })
        .on('mouseleave', () => {
          setTooltip({ x: 0, y: 0, content: null });
        });
    });

    // Legend
    const legend = g.append('g')
      .attr('transform', `translate(${innerWidth + 10}, 0)`);

    principles.forEach((principle, i) => {
      const legendRow = legend.append('g')
        .attr('transform', `translate(0, ${i * 20})`)
        .style('cursor', 'pointer')
        .on('mouseenter', () => setHoveredPrinciple(principle))
        .on('mouseleave', () => setHoveredPrinciple(null));

      legendRow.append('rect')
        .attr('width', 10)
        .attr('height', 10)
        .attr('fill', colorScale(principle));

      legendRow.append('text')
        .attr('x', 15)
        .attr('y', 8)
        .style('font-size', '12px')
        .text(principle);
    });

  }, [data, selectedPrinciples, hoveredPrinciple, width, height]);

  return (
    <div className="relative">
      <svg ref={svgRef}></svg>
      {tooltip.content && (
        <div
          className="absolute bg-white p-2 rounded shadow-lg border border-gray-200 text-sm z-10"
          style={{ left: tooltip.x + 10, top: tooltip.y - 10 }}
        >
          <div className="font-semibold">{tooltip.content.principle}</div>
          <div>Strength: {(tooltip.content.strength * 100).toFixed(1)}%</div>
          <div>Consistency: {(tooltip.content.consistency * 100).toFixed(1)}%</div>
          <div className="text-gray-500">
            {new Date(tooltip.content.timestamp).toLocaleTimeString()}
          </div>
        </div>
      )}
    </div>
  );
};
