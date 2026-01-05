import json
import pandas as pd
from shapely import wkt
from datetime import datetime
from typing import Dict
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_network_from_csv(csv_file: str) -> Dict:
    """Load network data from CSV file"""
    print(f"Loading network data:  {csv_file}")
    
    df = pd.read_csv(csv_file)
    print(f"  ✓ Loaded {len(df)} edge records")
    
    df['geometry'] = df['geometry'].apply(wkt.loads)
    
    nodes_dict = {}
    edges = []
    
    for idx, row in df.iterrows():
        geom = row['geometry']
        coords = list(geom.coords)
        
        from_node = str(int(row['from_node']))
        to_node = str(int(row['to_node']))
        
        if from_node not in nodes_dict:
            nodes_dict[from_node] = {
                'id': from_node,
                'lon': coords[0][0],
                'lat': coords[0][1],
                'degree_in': 0,
                'degree_out': 0
            }
        nodes_dict[from_node]['degree_out'] += 1
        
        if to_node not in nodes_dict:
            nodes_dict[to_node] = {
                'id': to_node,
                'lon': coords[-1][0],
                'lat': coords[-1][1],
                'degree_in': 0,
                'degree_out':  0
            }
        nodes_dict[to_node]['degree_in'] += 1
        
        edge_coords = [[lat, lon] for lon, lat in coords]
        
        edges.append({
            'from':  from_node,
            'to': to_node,
            'from_coords': edge_coords[0],
            'to_coords': edge_coords[-1],
            'path_coords': edge_coords,
            'length': float(row['len']) if 'len' in row else 0,
            'lanes': int(row['nlane']) if 'nlane' in row and pd.notna(row['nlane']) else 0,
            'road_id': str(row['cid']) if 'cid' in row else ''
        })
    
    nodes = []
    for node_id, node_data in nodes_dict.items():
        total_deg = node_data['degree_in'] + node_data['degree_out']
        
        node_type = 'normal'
        if total_deg == 1:
            node_type = 'terminal'
        elif total_deg >= 4:
            node_type = 'hub'
        
        nodes.append({
            'id':  node_id,
            'lat': float(node_data['lat']),
            'lon': float(node_data['lon']),
            'degree_in': int(node_data['degree_in']),
            'degree_out': int(node_data['degree_out']),
            'degree_total': int(total_deg),
            'type': node_type
        })
    
    degrees = [n['degree_total'] for n in nodes]
    lengths = [e['length'] for e in edges]
    
    stats = {
        'num_nodes': len(nodes),
        'num_edges': len(edges),
        'avg_degree': float(np.mean(degrees)) if degrees else 0,
        'max_degree': int(max(degrees)) if degrees else 0,
        'avg_edge_length': float(np.mean(lengths)) if lengths else 0,
        'total_length': float(sum(lengths)) if lengths else 0,
        'num_terminals': sum(1 for n in nodes if n['type'] == 'terminal'),
        'num_hubs': sum(1 for n in nodes if n['type'] == 'hub')
    }
    
    print(f"  ✓ Extraction complete:  {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    
    lats = [n['lat'] for n in nodes]
    lons = [n['lon'] for n in nodes]
    print(f"  ✓ Coordinate range:")
    print(f"    Latitude: {min(lats):.6f} ~ {max(lats):.6f}")
    print(f"    Longitude: {min(lons):.6f} ~ {max(lons):.6f}")
    
    return {
        'nodes': nodes,
        'edges': edges,
        'stats': stats
    }


def generate_network_html_from_csv(
    csv_file: str,
    output_file: str = 'network_visualization.html',
    title: str = 'Road Network Visualization'
) -> None:
    """
    Generate network visualization HTML from CSV file
    
    Args:
        csv_file: CSV file path
        output_file: Output HTML file path
        title: Page title
    """
    print(f"\n{'='*70}")
    print(f"Generating Network Visualization HTML")
    print(f"{'='*70}")
    
    network_data = load_network_from_csv(csv_file)
    
    data_json = {
        'network': network_data,
        'settings': {'enable_clustering': False}
    }
    
    html_content = _generate_topology_html(data_json, title)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n  ✓ Network visualization HTML generated: {output_file}")
    print(f"  - Nodes: {network_data['stats']['num_nodes']}")
    print(f"  - Edges: {network_data['stats']['num_edges']}")
    print(f"  - Avg Degree: {network_data['stats']['avg_degree']:.2f}")
    print(f"  - Total Length: {network_data['stats']['total_length']/1000:.2f} km")
    print(f"{'='*70}\n")


def _generate_topology_html(data_json: Dict, title: str) -> str:
    """Generate complete HTML content with English basemap"""
    
    html = f'''<! DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
          integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
          crossorigin=""/>
    
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Arial', 'Helvetica', sans-serif;
            background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }}
        
        .container {{ max-width: 1800px; margin: 0 auto; }}
        
        header {{
            background: white;
            border-radius: 15px;
            padding: 25px 30px;
            margin-bottom: 20px;
            box-shadow:  0 10px 30px rgba(0,0,0,0.15);
        }}
        
        h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .subtitle {{ color: #666; font-size: 1.05em; margin-top: 8px; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat-box {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 18px;
            border-radius:  12px;
            text-align:  center;
            transition: transform 0.2s;
        }}
        
        .stat-box:hover {{ transform: translateY(-3px); }}
        .stat-label {{ font-size: 0.9em; color: #666; margin-bottom: 8px; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: #667eea; }}
        .stat-unit {{ font-size: 0.6em; color: #999; }}
        
        .card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            margin-bottom:  20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        
        .controls {{
            display: flex;
            gap: 15px;
            flex-wrap: wrap;
            margin-bottom: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 15px;
            background: white;
            border-radius: 8px;
        }}
        
        .control-group label {{ font-weight: 600; cursor: pointer; }}
        
        input[type="checkbox"] {{ width: 20px; height: 20px; cursor: pointer; }}
        input[type="range"] {{ width:  150px; cursor: pointer; }}
        
        .btn {{
            padding: 12px 25px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            font-size: 1em;
            font-weight:  600;
            transition: all 0.3s;
            color: white;
        }}
        
        .btn-primary {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .btn-primary:hover {{ transform: translateY(-2px); }}
        .btn-secondary {{ background: #28a745; }}
        .btn-danger {{ background: #dc3545; }}
        
        #map {{
            height: 750px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            border: 3px solid #e0e0e0;
            background: #f0f0f0;
        }}
        
        .legend {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-top:  20px;
        }}
        
        .legend-title {{
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 8px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom:  8px;
        }}
        
        .legend-circle {{ width: 14px; height: 14px; border-radius: 50%; }}
        .legend-color {{ width: 35px; height: 3px; border-radius: 2px; }}
        
        .loading {{
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            padding: 30px 50px;
            border-radius:  15px;
            box-shadow:  0 10px 40px rgba(0,0,0,0.3);
            z-index: 10000;
            display: none;
        }}
        
        .loading.active {{ display: block; }}
        
        .spinner {{
            border: 4px solid #f3f3f3;
            border-top:  4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .loading-text {{ text-align: center; font-weight: 600; }}
        
        .leaflet-popup-content {{ font-size: 0.9em; min-width: 200px; }}
        .popup-title {{
            font-size: 1.1em;
            font-weight:  bold;
            color: #667eea;
            margin-bottom: 10px;
            border-bottom: 2px solid #667eea;
            padding-bottom: 5px;
        }}
        .info-table {{ width: 100%; font-size: 0.9em; }}
        .info-table td {{ padding: 4px 8px; }}
        .info-table td: first-child {{ font-weight: 600; color: #666; }}
    </style>
</head>
<body>
    <div class="loading" id="loadingIndicator">
        <div class="spinner"></div>
        <div class="loading-text">Rendering network...</div>
    </div>
    
    <div class="container">
        <header>
            <h1>🗺️ {title}</h1>
            <p class="subtitle">Interactive Road Network Visualization with Leaflet + OpenStreetMap</p>
            <p class="subtitle">Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">📍 Total Nodes</div>
                    <div class="stat-value" id="nodeCount">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">🔗 Total Edges</div>
                    <div class="stat-value" id="edgeCount">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">📊 Avg Degree</div>
                    <div class="stat-value" id="avgDegree">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">🔝 Max Degree</div>
                    <div class="stat-value" id="maxDegree">-</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">📏 Total Length</div>
                    <div class="stat-value" id="totalLength">-<span class="stat-unit">km</span></div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">🎯 Hub Nodes</div>
                    <div class="stat-value" id="hubCount">-</div>
                </div>
            </div>
        </header>
        
        <div class="card">
            <div class="controls">
                <div class="control-group">
                    <input type="checkbox" id="toggleNodes" checked>
                    <label for="toggleNodes">Show Nodes</label>
                </div>
                
                <div class="control-group">
                    <input type="checkbox" id="toggleEdges" checked>
                    <label for="toggleEdges">Show Edges</label>
                </div>
                
                <div class="control-group">
                    <input type="checkbox" id="colorByDegree" checked>
                    <label for="colorByDegree">Color by Degree</label>
                </div>
                
                <div class="control-group">
                    <label for="edgeOpacity">Edge Opacity:  </label>
                    <input type="range" id="edgeOpacity" min="0" max="100" value="60" step="5">
                    <span id="edgeOpacityValue">60%</span>
                </div>
                
                <div class="control-group">
                    <label for="nodeSize">Node Size: </label>
                    <input type="range" id="nodeSize" min="1" max="10" value="4" step="1">
                    <span id="nodeSizeValue">4</span>
                </div>
                
                <button class="btn btn-primary" onclick="exportAsSVGWithMap()">
                    💾 Export SVG (with basemap)
                </button>
                
                <button class="btn btn-secondary" onclick="exportAsSVGNoMap()">
                    📄 Export SVG (network only)
                </button>
                
                <button class="btn btn-danger" onclick="resetView()">
                    🔄 Reset View
                </button>
            </div>
            
            <div id="map"></div>
            
            <div class="legend">
                <div class="legend-title">📖 Legend</div>
                <div class="legend-item">
                    <div class="legend-circle" style="background: #4285F4; border:  2px solid #1a73e8;"></div>
                    <span>Normal Node (degree 2-3)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-circle" style="background: #FBBC04; border: 2px solid #F9AB00;"></div>
                    <span>Terminal Node (degree 1)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-circle" style="background: #EA4335; border: 2px solid #C5221F;"></div>
                    <span>Hub Node (degree ≥ 4)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #666; height: 2px;"></div>
                    <span>Road Link</span>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
            integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo="
            crossorigin=""></script>
    
    <script>
        window.addEventListener('load', function() {{
            if (typeof L === 'undefined') {{
                alert('❌ Leaflet library failed to load');
                return;
            }}
            
            const data = {json.dumps(data_json, ensure_ascii=False, cls=NumpyEncoder)};
            
            let map, edgeLayer, nodeLayer, bounds;
            let currentEdgeOpacity = 0.6, currentNodeSize = 4;
            
            showLoading();
            setTimeout(() => {{
                try {{
                    initMap();
                    updateStats();
                    setupEventListeners();
                    hideLoading();
                }} catch(e) {{
                    console.error('❌ Error:', e);
                    hideLoading();
                    alert('Map loading failed:  ' + e.message);
                }}
            }}, 500);
            
            function showLoading() {{
                document.getElementById('loadingIndicator').classList.add('active');
            }}
            
            function hideLoading() {{
                document.getElementById('loadingIndicator').classList.remove('active');
            }}
            
            function initMap() {{
                if (! data.network.nodes || data.network.nodes.length === 0) {{
                    throw new Error('No node data');
                }}
                
                const lats = data.network.nodes.map(n => n.lat);
                const lons = data.network.nodes.map(n => n.lon);
                const minLat = Math.min(...lats);
                const maxLat = Math.max(...lats);
                const minLon = Math.min(...lons);
                const maxLon = Math.max(...lons);
                
                bounds = [[minLat, minLon], [maxLat, maxLon]];
                const center = [(minLat + maxLat) / 2, (minLon + maxLon) / 2];
                
                map = L.map('map', {{
                    center: center,
                    zoom: 13,
                    zoomControl: true,
                    preferCanvas: true
                }});
                
                // Use English version of OpenStreetMap
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                    attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                    maxZoom: 19
                }}).addTo(map);
                
                edgeLayer = L.layerGroup().addTo(map);
                nodeLayer = L.layerGroup().addTo(map);
                
                drawEdges();
                drawNodes();
                
                map.fitBounds(bounds, {{ padding: [50, 50] }});
            }}
            
            function drawEdges() {{
                edgeLayer.clearLayers();
                
                data.network.edges.forEach(edge => {{
                    const coords = edge.path_coords || [edge.from_coords, edge.to_coords];
                    
                    const polyline = L.polyline(coords, {{
                        color: '#666',
                        weight: 2,
                        opacity: currentEdgeOpacity
                    }});
                    
                    polyline.bindPopup(`
                        <div class="popup-title">🔗 Road Information</div>
                        <table class="info-table">
                            <tr><td>From:</td><td>${{edge.from}}</td></tr>
                            <tr><td>To:</td><td>${{edge.to}}</td></tr>
                            <tr><td>Length:</td><td>${{edge.length.toFixed(2)}} m</td></tr>
                            <tr><td>Lanes:</td><td>${{edge.lanes}}</td></tr>
                        </table>
                    `);
                    polyline.addTo(edgeLayer);
                }});
            }}
            
            function drawNodes() {{
                nodeLayer.clearLayers();
                
                const colorByDegree = document.getElementById('colorByDegree').checked;
                
                data.network.nodes.forEach(node => {{
                    let fillColor, strokeColor, radius;
                    
                    if (colorByDegree) {{
                        if (node.type === 'terminal') {{
                            fillColor = '#FBBC04';
                            strokeColor = '#F9AB00';
                            radius = currentNodeSize;
                        }} else if (node.type === 'hub') {{
                            fillColor = '#EA4335';
                            strokeColor = '#C5221F';
                            radius = currentNodeSize + 2;
                        }} else {{
                            fillColor = '#4285F4';
                            strokeColor = '#1a73e8';
                            radius = currentNodeSize;
                        }}
                    }} else {{
                        fillColor = '#4285F4';
                        strokeColor = '#1a73e8';
                        radius = currentNodeSize;
                    }}
                    
                    const circle = L.circleMarker([node.lat, node.lon], {{
                        radius: radius,
                        fillColor: fillColor,
                        color: strokeColor,
                        weight: 2,
                        fillOpacity: 0.8
                    }});
                    
                    circle.bindPopup(`
                        <div class="popup-title">📍 Node ${{node.id}}</div>
                        <table class="info-table">
                            <tr><td>Coordinates:</td><td>${{node.lat.toFixed(6)}}, ${{node.lon.toFixed(6)}}</td></tr>
                            <tr><td>In-degree:</td><td>${{node.degree_in}}</td></tr>
                            <tr><td>Out-degree:</td><td>${{node.degree_out}}</td></tr>
                        </table>
                    `);
                    circle.addTo(nodeLayer);
                }});
            }}
            
            function setupEventListeners() {{
                document.getElementById('toggleNodes').addEventListener('change', function(e) {{
                    e.target.checked ? map.addLayer(nodeLayer) : map.removeLayer(nodeLayer);
                }});
                
                document.getElementById('toggleEdges').addEventListener('change', function(e) {{
                    e.target.checked ? map.addLayer(edgeLayer) : map.removeLayer(edgeLayer);
                }});
                
                document.getElementById('colorByDegree').addEventListener('change', () => drawNodes());
                
                document.getElementById('edgeOpacity').addEventListener('input', function(e) {{
                    currentEdgeOpacity = e.target.value / 100;
                    document.getElementById('edgeOpacityValue').textContent = e.target.value + '%';
                    drawEdges();
                }});
                
                document.getElementById('nodeSize').addEventListener('input', function(e) {{
                    currentNodeSize = parseInt(e.target.value);
                    document.getElementById('nodeSizeValue').textContent = e.target.value;
                    drawNodes();
                }});
            }}
            
            function updateStats() {{
                document.getElementById('nodeCount').textContent = data.network.stats.num_nodes;
                document.getElementById('edgeCount').textContent = data.network.stats.num_edges;
                document.getElementById('avgDegree').textContent = data.network.stats.avg_degree.toFixed(2);
                document.getElementById('maxDegree').textContent = data.network.stats.max_degree;
                document.getElementById('totalLength').innerHTML = 
                    (data.network.stats.total_length / 1000).toFixed(2) + '<span class="stat-unit">km</span>';
                document.getElementById('hubCount').textContent = data.network.stats.num_hubs;
            }}
            
            window.resetView = function() {{
                map.fitBounds(bounds, {{ padding:  [50, 50] }});
            }}
            
            // Export SVG with basemap
            window.exportAsSVGWithMap = async function() {{
                showLoading();
                document.getElementById('loadingIndicator').querySelector('.loading-text').textContent = 'Capturing basemap...';
                
                try {{
                    if (typeof html2canvas === 'undefined') {{
                        await loadScript('https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js');
                    }}
                    
                    const mapElement = document.getElementById('map');
                    const canvas = await html2canvas(mapElement, {{
                        useCORS: true,
                        allowTaint: false,
                        logging: false
                    }});
                    
                    document.getElementById('loadingIndicator').querySelector('.loading-text').textContent = 'Generating SVG...';
                    
                    const mapImageData = canvas.toDataURL('image/png');
                    const size = map.getSize();
                    
                    let svg = '<?xml version="1.0" encoding="UTF-8"?>\\n';
                    svg += '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" ';
                    svg += `width="${{size.x}}" height="${{size.y}}" viewBox="0 0 ${{size.x}} ${{size.y}}">\\n`;
                    svg += `  <image xlink:href="${{mapImageData}}" width="${{size.x}}" height="${{size.y}}"/>\\n`;
                    
                    svg += `  <g id="edges">\\n`;
                    data.network.edges.forEach(edge => {{
                        const coords = edge.path_coords || [edge.from_coords, edge.to_coords];
                        let pathData = 'M';
                        coords.forEach((coord, i) => {{
                            const point = map.latLngToContainerPoint(coord);
                            pathData += i === 0 ? `${{point.x}},${{point.y}}` : ` L${{point.x}},${{point.y}}`;
                        }});
                        svg += `    <path d="${{pathData}}" stroke="#666" stroke-width="2" fill="none" opacity="${{currentEdgeOpacity}}"/>\\n`;
                    }});
                    svg += `  </g>\\n`;
                    
                    svg += `  <g id="nodes">\\n`;
                    const colorByDegree = document.getElementById('colorByDegree').checked;
                    data.network.nodes.forEach(node => {{
                        const point = map.latLngToContainerPoint([node.lat, node.lon]);
                        let fillColor, strokeColor, radius;
                        
                        if (colorByDegree) {{
                            if (node.type === 'terminal') {{
                                fillColor = '#FBBC04'; strokeColor = '#F9AB00'; radius = currentNodeSize;
                            }} else if (node.type === 'hub') {{
                                fillColor = '#EA4335'; strokeColor = '#C5221F'; radius = currentNodeSize + 2;
                            }} else {{
                                fillColor = '#4285F4'; strokeColor = '#1a73e8'; radius = currentNodeSize;
                            }}
                        }} else {{
                            fillColor = '#4285F4'; strokeColor = '#1a73e8'; radius = currentNodeSize;
                        }}
                        
                        svg += `    <circle cx="${{point.x}}" cy="${{point.y}}" r="${{radius}}" fill="${{fillColor}}" stroke="${{strokeColor}}" stroke-width="2" opacity="0.8"/>\\n`;
                    }});
                    svg += `  </g>\\n`;
                    
                    svg += `  <text x="20" y="30" font-family="Arial" font-size="20" font-weight="bold" fill="#333" stroke="white" stroke-width="3" paint-order="stroke">{title}</text>\\n`;
                    svg += `</svg>`;
                    
                    downloadFile(svg, 'network_with_basemap.svg', 'image/svg+xml');
                    hideLoading();
                    alert('✅ SVG file (with basemap) exported successfully!');
                }} catch(e) {{
                    hideLoading();
                    alert('❌ Export failed: ' + e.message);
                }}
            }}
            
            // Export SVG (network only)
            window.exportAsSVGNoMap = function() {{
                showLoading();
                
                setTimeout(() => {{
                    const size = map.getSize();
                    
                    let svg = '<?xml version="1.0" encoding="UTF-8"? >' + '\\n';
                    svg += '<svg xmlns="http://www.w3.org/2000/svg" ';
                    svg += `width="${{size.x}}" height="${{size.y}}" viewBox="0 0 ${{size.x}} ${{size.y}}">\\n`;
                    svg += `  <rect width="100%" height="100%" fill="#f8f9fa"/>\\n`;
                    
                    svg += `  <g id="edges">\\n`;
                    data.network.edges.forEach(edge => {{
                        const coords = edge.path_coords || [edge.from_coords, edge.to_coords];
                        let pathData = 'M';
                        coords.forEach((coord, i) => {{
                            const point = map.latLngToContainerPoint(coord);
                            pathData += i === 0 ? `${{point.x}},${{point.y}}` : ` L${{point.x}},${{point.y}}`;
                        }});
                        svg += `    <path d="${{pathData}}" stroke="#666" stroke-width="2" fill="none" opacity="${{currentEdgeOpacity}}"/>\\n`;
                    }});
                    svg += `  </g>\\n`;
                    
                    svg += `  <g id="nodes">\\n`;
                    const colorByDegree = document.getElementById('colorByDegree').checked;
                    data.network.nodes.forEach(node => {{
                        const point = map.latLngToContainerPoint([node.lat, node.lon]);
                        let fillColor, strokeColor, radius;
                        
                        if (colorByDegree) {{
                            if (node.type === 'terminal') {{
                                fillColor = '#FBBC04'; strokeColor = '#F9AB00'; radius = currentNodeSize;
                            }} else if (node.type === 'hub') {{
                                fillColor = '#EA4335'; strokeColor = '#C5221F'; radius = currentNodeSize + 2;
                            }} else {{
                                fillColor = '#4285F4'; strokeColor = '#1a73e8'; radius = currentNodeSize;
                            }}
                        }} else {{
                            fillColor = '#4285F4'; strokeColor = '#1a73e8'; radius = currentNodeSize;
                        }}
                        
                        svg += `    <circle cx="${{point.x}}" cy="${{point.y}}" r="${{radius}}" fill="${{fillColor}}" stroke="${{strokeColor}}" stroke-width="2" opacity="0.8"/>\\n`;
                    }});
                    svg += `  </g>\\n`;
                    
                    svg += `  <text x="20" y="30" font-family="Arial" font-size="20" font-weight="bold" fill="#333">{title}</text>\\n`;
                    svg += `</svg>`;
                    
                    downloadFile(svg, 'network_no_basemap.svg', 'image/svg+xml');
                    hideLoading();
                    alert('✅ SVG file (network only) exported successfully!');
                }}, 200);
            }}
            
            function loadScript(url) {{
                return new Promise((resolve, reject) => {{
                    const script = document.createElement('script');
                    script.src = url;
                    script.onload = resolve;
                    script.onerror = reject;
                    document.head.appendChild(script);
                }});
            }}
            
            function downloadFile(content, filename, mimeType) {{
                const blob = new Blob([content], {{ type: mimeType + ';charset=utf-8' }});
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = filename;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
            }}
        }});
    </script>
</body>
</html>'''
    
    return html


# # Usage example
# if __name__ == "__main__":
#     generate_network_html_from_csv(
#         csv_file='base_data/xc_edge.csv',
#         output_file='xuancheng_network_en.html',
#         title='Xuancheng Road Network Visualization'
#     )
    
#     print("\n✅ Generation complete!")
#     print("📂 Please open in browser:  xuancheng_network_en.html")
# 使用示例
if __name__ == "__main__": 
    # 直接从CSV文件生成可视化
    generate_network_html_from_csv(
        csv_file='largest_connected_component.csv',  # 您的CSV文件路径
        output_file='xuancheng_network.html',
        title='宣城市路网可视化'
    )
    
    print("\n✅ 生成完成！")
    print("📂 请在浏览器中打开:  xuancheng_network.html")
    print("\n💡 特点:")
    print("  ✓ 直接从CSV加载，无需NetworkX")
    print("  ✓ 支持完整路径几何（多段线）")
    print("  ✓ 修复了Leaflet加载问题")
    print("  ✓ 显示道路详细信息（车道数、道路ID等）")