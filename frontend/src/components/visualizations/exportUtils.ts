import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

interface ExportOptions {
  format: 'png' | 'pdf' | 'svg';
  filename?: string;
  quality?: number;
}

/**
 * Export a visualization element as an image or PDF
 * @param element - The DOM element containing the visualization
 * @param options - Export options including format and filename
 */
export async function exportVisualization(
  element: HTMLElement,
  options: ExportOptions = { format: 'png' }
): Promise<void> {
  const { format, filename = `visualization-${Date.now()}`, quality = 0.95 } = options;

  try {
    switch (format) {
      case 'png':
        await exportAsPNG(element, filename, quality);
        break;
      case 'pdf':
        await exportAsPDF(element, filename, quality);
        break;
      case 'svg':
        await exportAsSVG(element, filename);
        break;
      default:
        throw new Error(`Unsupported export format: ${format}`);
    }
  } catch (error) {
    console.error('Export failed:', error);
    throw error;
  }
}

async function exportAsPNG(element: HTMLElement, filename: string, quality: number): Promise<void> {
  const canvas = await html2canvas(element, {
    backgroundColor: '#ffffff',
    scale: 2, // Higher resolution
    logging: false,
  });

  // Convert canvas to blob
  canvas.toBlob((blob) => {
    if (!blob) {
      throw new Error('Failed to create image blob');
    }

    // Create download link
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `${filename}.png`;
    link.click();

    // Clean up
    URL.revokeObjectURL(url);
  }, 'image/png', quality);
}

async function exportAsPDF(element: HTMLElement, filename: string, quality: number): Promise<void> {
  const canvas = await html2canvas(element, {
    backgroundColor: '#ffffff',
    scale: 2,
    logging: false,
  });

  // Calculate PDF dimensions
  const imgWidth = 210; // A4 width in mm
  const imgHeight = (canvas.height * imgWidth) / canvas.width;
  
  // Create PDF
  const pdf = new jsPDF({
    orientation: imgHeight > imgWidth ? 'portrait' : 'landscape',
    unit: 'mm',
  });

  const pageHeight = pdf.internal.pageSize.height;
  let heightLeft = imgHeight;
  let position = 0;

  // Add image to PDF
  pdf.addImage(
    canvas.toDataURL('image/png', quality),
    'PNG',
    0,
    position,
    imgWidth,
    imgHeight
  );

  // Add additional pages if needed
  while (heightLeft > pageHeight) {
    position = heightLeft - pageHeight;
    pdf.addPage();
    pdf.addImage(
      canvas.toDataURL('image/png', quality),
      'PNG',
      0,
      -position,
      imgWidth,
      imgHeight
    );
    heightLeft -= pageHeight;
  }

  // Save PDF
  pdf.save(`${filename}.pdf`);
}

async function exportAsSVG(element: HTMLElement, filename: string): Promise<void> {
  // Check if element contains an SVG
  const svg = element.querySelector('svg');
  if (!svg) {
    throw new Error('No SVG element found in the visualization');
  }

  // Clone the SVG to avoid modifying the original
  const svgClone = svg.cloneNode(true) as SVGElement;
  
  // Add necessary attributes
  svgClone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  svgClone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');

  // Convert to string
  const svgString = new XMLSerializer().serializeToString(svgClone);
  
  // Create blob
  const blob = new Blob([svgString], { type: 'image/svg+xml' });
  
  // Create download link
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.svg`;
  link.click();

  // Clean up
  URL.revokeObjectURL(url);
}

/**
 * Export multiple visualizations as a combined report
 * @param elements - Array of DOM elements containing visualizations
 * @param title - Report title
 * @param format - Export format (currently only PDF supported for multiple visualizations)
 */
export async function exportVisualizationReport(
  elements: HTMLElement[],
  title: string = 'Visualization Report',
  format: 'pdf' = 'pdf'
): Promise<void> {
  if (format !== 'pdf') {
    throw new Error('Only PDF format is supported for reports');
  }

  const pdf = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  // Add title page
  pdf.setFontSize(24);
  pdf.text(title, 105, 40, { align: 'center' });
  pdf.setFontSize(12);
  pdf.text(new Date().toLocaleDateString(), 105, 60, { align: 'center' });

  // Add each visualization
  for (let i = 0; i < elements.length; i++) {
    if (i > 0) {
      pdf.addPage();
    }

    const canvas = await html2canvas(elements[i], {
      backgroundColor: '#ffffff',
      scale: 2,
      logging: false,
    });

    const imgWidth = 190; // Leave margins
    const imgHeight = (canvas.height * imgWidth) / canvas.width;
    
    // Center the image on the page
    const x = (210 - imgWidth) / 2;
    const y = 20;

    pdf.addImage(
      canvas.toDataURL('image/png'),
      'PNG',
      x,
      y,
      imgWidth,
      Math.min(imgHeight, 250) // Limit height to fit on page
    );
  }

  // Save the report
  pdf.save(`${title.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}.pdf`);
}

/**
 * Prepare data for export in various formats
 * @param data - The raw data to export
 * @param format - Export format
 */
export function exportData(data: any, format: 'json' | 'csv', filename: string = 'data'): void {
  let content: string;
  let mimeType: string;
  let extension: string;

  switch (format) {
    case 'json':
      content = JSON.stringify(data, null, 2);
      mimeType = 'application/json';
      extension = 'json';
      break;
    case 'csv':
      content = convertToCSV(data);
      mimeType = 'text/csv';
      extension = 'csv';
      break;
    default:
      throw new Error(`Unsupported data export format: ${format}`);
  }

  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${filename}.${extension}`;
  link.click();
  URL.revokeObjectURL(url);
}

function convertToCSV(data: any[]): string {
  if (!Array.isArray(data) || data.length === 0) {
    return '';
  }

  // Get headers from first object
  const headers = Object.keys(data[0]);
  const csvHeaders = headers.join(',');

  // Convert each object to CSV row
  const csvRows = data.map(row => {
    return headers.map(header => {
      const value = row[header];
      // Escape values containing commas or quotes
      if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
        return `"${value.replace(/"/g, '""')}"`;
      }
      return value;
    }).join(',');
  });

  return [csvHeaders, ...csvRows].join('\n');
}
