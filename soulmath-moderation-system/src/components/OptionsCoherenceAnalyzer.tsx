import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, Cell, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { TrendingUp, TrendingDown, AlertTriangle, CheckCircle, Activity, Brain, Eye, Zap, Target, BarChart3 } from 'lucide-react';

const OptionsCoherenceAnalyzer: React.FC = () => {
  const [selectedStock, setSelectedStock] = useState('FBK');
  const [analysisResults, setAnalysisResults] = useState<any>(null);
  const [optionsData, setOptionsData] = useState<any>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [coherenceHistory, setCoherenceHistory] = useState<any[]>([]);
  const [alerts, setAlerts] = useState<any[]>([]);

  // Sample options data for different stocks
  const sampleOptionsData: Record<string, any> = {
    'FBK': {
      symbol: "FBK", 
      currentPrice: 48.22,
      expirationDate: "2025-01-16",
      timestamp: new Date().toISOString(),
      calls: [
        { strike: 45, bid: 3.00, ask: 3.70, impliedVol: 0.28, delta: 0.65, gamma: 0.03, volume: 150, openInterest: 1200 },
        { strike: 50, bid: 1.35, ask: 1.60, impliedVol: 0.32, delta: 0.45, gamma: 0.04, volume: 280, openInterest: 2100 },
        { strike: 55, bid: 0.60, ask: 0.85, impliedVol: 0.35, delta: 0.25, gamma: 0.03, volume: 90, openInterest: 800 }
      ],
      puts: [
        { strike: 45, bid: 0.85, ask: 1.10, impliedVol: 0.30, delta: -0.35, gamma: 0.03, volume: 120, openInterest: 900 },
        { strike: 50, bid: 1.80, ask: 2.10, impliedVol: 0.33, delta: -0.55, gamma: 0.04, volume: 350, openInterest: 1800 },
        { strike: 55, bid: 4.20, ask: 4.80, impliedVol: 0.36, delta: -0.75, gamma: 0.03, volume: 200, openInterest: 1100 }
      ]
    },
    'AAPL': {
      symbol: "AAPL",
      currentPrice: 212.42,
      expirationDate: "2025-01-17",
      timestamp: new Date().toISOString(),
      calls: [
        { strike: 210, bid: 8.50, ask: 9.20, impliedVol: 0.22, delta: 0.58, gamma: 0.02, volume: 3200, openInterest: 15000 },
        { strike: 215, bid: 5.20, ask: 5.80, impliedVol: 0.24, delta: 0.41, gamma: 0.025, volume: 2800, openInterest: 12000 },
        { strike: 220, bid: 2.80, ask: 3.20, impliedVol: 0.26, delta: 0.28, gamma: 0.022, volume: 1900, openInterest: 8500 }
      ],
      puts: [
        { strike: 210, bid: 5.80, ask: 6.40, impliedVol: 0.23, delta: -0.42, gamma: 0.02, volume: 2100, openInterest: 11000 },
        { strike: 215, bid: 8.10, ask: 8.70, impliedVol: 0.25, delta: -0.59, gamma: 0.025, volume: 3100, openInterest: 14000 },
        { strike: 220, bid: 10.90, ask: 11.60, impliedVol: 0.27, delta: -0.72, gamma: 0.022, volume: 1600, openInterest: 7200 }
      ]
    },
    'TSLA': {
      symbol: "TSLA",
      currentPrice: 309.93,
      expirationDate: "2025-01-17", 
      timestamp: new Date().toISOString(),
      calls: [
        { strike: 300, bid: 18.50, ask: 22.80, impliedVol: 0.68, delta: 0.61, gamma: 0.008, volume: 450, openInterest: 2800 },
        { strike: 310, bid: 12.20, ask: 16.10, impliedVol: 0.71, delta: 0.48, gamma: 0.009, volume: 380, openInterest: 2200 },
        { strike: 320, bid: 7.80, ask: 11.40, impliedVol: 0.74, delta: 0.35, gamma: 0.008, volume: 290, openInterest: 1600 }
      ],
      puts: [
        { strike: 300, bid: 8.90, ask: 12.20, impliedVol: 0.69, delta: -0.39, gamma: 0.008, volume: 320, openInterest: 1900 },
        { strike: 310, bid: 12.40, ask: 16.80, impliedVol: 0.72, delta: -0.52, gamma: 0.009, volume: 410, openInterest: 2500 },
        { strike: 320, bid: 17.20, ask: 21.90, impliedVol: 0.75, delta: -0.65, gamma: 0.008, volume: 280, openInterest: 1800 }
      ]
    }
  };

  // GCT Options Coherence Analysis Engine
  const analyzeOptionsCoherence = async (optionsData: any) => {
    const { calls, puts, currentPrice, symbol } = optionsData;
    
    // Enhanced ψ (Psi) - Pricing Structure Clarity
    const calculatePsi = (calls: any[], puts: any[], currentPrice: number) => {
      let clarityScore = 0;
      let totalChecks = 0;
      
      // 1. Monotonicity check for calls
      for (let i = 0; i < calls.length - 1; i++) {
        totalChecks++;
        const current = calls[i];
        const next = calls[i + 1];
        
        if (next.strike > current.strike) {
          const currentMid = (current.bid + current.ask) / 2;
          const nextMid = (next.bid + next.ask) / 2;
          
          if (nextMid <= currentMid) {
            clarityScore += 1;
          } else {
            clarityScore -= 0.5; // Penalty for price inversion
          }
        }
      }
      
      // 2. Put-call parity relationships
      for (let i = 0; i < Math.min(calls.length, puts.length); i++) {
        if (calls[i].strike === puts[i].strike) {
          totalChecks++;
          const callMid = (calls[i].bid + calls[i].ask) / 2;
          const putMid = (puts[i].bid + puts[i].ask) / 2;
          const strike = calls[i].strike;
          
          // Simplified put-call parity: C - P ≈ S - K (ignoring interest/dividends)
          const leftSide = callMid - putMid;
          const rightSide = currentPrice - strike;
          const parityError = Math.abs(leftSide - rightSide) / currentPrice;
          
          if (parityError < 0.05) {
            clarityScore += 1;
          } else if (parityError > 0.15) {
            clarityScore -= 0.3;
          }
        }
      }
      
      // 3. Bid-ask spread analysis
      const allOptions = [...calls, ...puts];
      allOptions.forEach(opt => {
        totalChecks++;
        const spread = opt.ask - opt.bid;
        const midPrice = (opt.ask + opt.bid) / 2;
        const spreadRatio = spread / midPrice;
        
        if (spreadRatio < 0.08) {
          clarityScore += 1; // Tight spreads indicate clarity
        } else if (spreadRatio > 0.25) {
          clarityScore -= 0.5; // Wide spreads reduce clarity
        }
      });
      
      return Math.max(0, Math.min(1, clarityScore / totalChecks));
    };
    
    // Enhanced ρ (Rho) - Market Wisdom & Experience
    const calculateRho = (calls: any[], puts: any[]) => {
      let wisdomScore = 0;
      let factors = 0;
      
      // 1. Implied Volatility Smile Consistency
      const callIVs = calls.map(c => c.impliedVol);
      const putIVs = puts.map(p => p.impliedVol);
      
      // Check for reasonable vol smile shape
      const callIVStd = Math.sqrt(callIVs.reduce((sum, iv) => sum + Math.pow(iv - callIVs.reduce((a,b) => a+b)/callIVs.length, 2), 0) / callIVs.length);
      const putIVStd = Math.sqrt(putIVs.reduce((sum, iv) => sum + Math.pow(iv - putIVs.reduce((a,b) => a+b)/putIVs.length, 2), 0) / putIVs.length);
      
      // Moderate volatility dispersion indicates experience
      if (callIVStd > 0.02 && callIVStd < 0.15) {
        wisdomScore += 0.25;
      }
      if (putIVStd > 0.02 && putIVStd < 0.15) {
        wisdomScore += 0.25;
      }
      factors += 0.5;
      
      // 2. Greeks Consistency
      const callDeltas = calls.map(c => c.delta);
      const putDeltas = puts.map(p => Math.abs(p.delta));
      
      // Delta should be monotonic
      let callDeltaConsistent = true;
      for (let i = 0; i < callDeltas.length - 1; i++) {
        if (callDeltas[i] <= callDeltas[i + 1]) {
          callDeltaConsistent = false;
        }
      }
      
      if (callDeltaConsistent) {
        wisdomScore += 0.3;
      }
      factors += 0.3;
      
      // 3. Volume-Open Interest Relationship
      const allOptions = [...calls, ...puts];
      let volumeOIRelationship = 0;
      allOptions.forEach(opt => {
        if (opt.openInterest > 0) {
          const volumeRatio = opt.volume / opt.openInterest;
          if (volumeRatio > 0.02 && volumeRatio < 0.5) {
            volumeOIRelationship += 1;
          }
        }
      });
      
      wisdomScore += (volumeOIRelationship / allOptions.length) * 0.2;
      factors += 0.2;
      
      return Math.min(1, wisdomScore / factors);
    };
    
    // Enhanced q (Emotional Charge) - Fear & Greed Indicators
    const calculateQ = (calls: any[], puts: any[], currentPrice: number) => {
      let emotionScore = 0;
      
      // 1. Put-Call Volume Ratio
      const totalCallVolume = calls.reduce((sum, c) => sum + c.volume, 0);
      const totalPutVolume = puts.reduce((sum, p) => sum + p.volume, 0);
      const putCallVolumeRatio = totalPutVolume / Math.max(totalCallVolume, 1);
      
      // Extreme ratios indicate high emotion
      if (putCallVolumeRatio > 1.5) {
        emotionScore += 0.3; // High fear
      } else if (putCallVolumeRatio < 0.5) {
        emotionScore += 0.25; // High greed
      }
      
      // 2. Implied Volatility Levels
      const allIVs = [...calls.map(c => c.impliedVol), ...puts.map(p => p.impliedVol)];
      const avgIV = allIVs.reduce((sum, iv) => sum + iv, 0) / allIVs.length;
      
      // High IV indicates uncertainty/emotion
      if (avgIV > 0.4) {
        emotionScore += 0.2;
      }
      if (avgIV > 0.6) {
        emotionScore += 0.3;
      }
      
      // 3. Skew Analysis (put vs call IV)
      const avgCallIV = calls.reduce((sum, c) => sum + c.impliedVol, 0) / calls.length;
      const avgPutIV = puts.reduce((sum, p) => sum + p.impliedVol, 0) / puts.length;
      const skew = avgPutIV - avgCallIV;
      
      // High put skew indicates fear
      if (Math.abs(skew) > 0.1) {
        emotionScore += 0.2;
      }
      
      return Math.min(1, emotionScore);
    };
    
    // Enhanced f (Social Consensus) - Market Agreement
    const calculateF = (calls: any[], puts: any[]) => {
      let consensusScore = 0;
      let factors = 0;
      
      // 1. Volume-weighted spread analysis
      const allOptions = [...calls, ...puts];
      let weightedSpreadSum = 0;
      let totalVolumeWeight = 0;
      
      allOptions.forEach(opt => {
        const spread = opt.ask - opt.bid;
        const midPrice = (opt.ask + opt.bid) / 2;
        const spreadRatio = spread / midPrice;
        const weight = opt.volume + opt.openInterest * 0.1;
        
        weightedSpreadSum += spreadRatio * weight;
        totalVolumeWeight += weight;
      });
      
      const avgWeightedSpread = totalVolumeWeight > 0 ? weightedSpreadSum / totalVolumeWeight : 0.5;
      consensusScore += Math.max(0, 1 - avgWeightedSpread * 4) * 0.4;
      factors += 0.4;
      
      // 2. Volume concentration
      const totalVolume = allOptions.reduce((sum, opt) => sum + opt.volume, 0);
      const maxVolume = Math.max(...allOptions.map(opt => opt.volume));
      const volumeConcentration = maxVolume / Math.max(totalVolume, 1);
      
      // Moderate concentration indicates good consensus
      if (volumeConcentration > 0.2 && volumeConcentration < 0.6) {
        consensusScore += 0.3;
      }
      factors += 0.3;
      
      // 3. Open Interest Distribution
      const totalOI = allOptions.reduce((sum, opt) => sum + opt.openInterest, 0);
      const oiDistribution = allOptions.map(opt => opt.openInterest / Math.max(totalOI, 1));
      const oiEntropy = -oiDistribution.reduce((sum, p) => p > 0 ? sum + p * Math.log2(p) : sum, 0);
      
      // Higher entropy indicates better consensus distribution
      consensusScore += Math.min(1, oiEntropy / 3) * 0.3;
      factors += 0.3;
      
      return Math.min(1, consensusScore / factors);
    };
    
    // Calculate core GCT variables
    const psi = calculatePsi(calls, puts, currentPrice);
    const rho = calculateRho(calls, puts);
    const q = calculateQ(calls, puts, currentPrice);
    const f = calculateF(calls, puts);
    
    // Enhanced coherence calculation with dynamic weighting
    const volatilityAdjustment = 1 - Math.min(0.3, q);
    const experienceWeight = 0.2 + (rho * 0.2);
    
    const coherence = (
      psi * 0.3 + 
      rho * experienceWeight + 
      (1 - q) * 0.2 * volatilityAdjustment + 
      f * 0.3
    );
    
    // Anomaly detection
    const anomalies: string[] = [];
    
    if (psi < 0.3) anomalies.push("Price structure inconsistencies detected");
    if (q > 0.7) anomalies.push("High emotional distortion in pricing");
    if (f < 0.3) anomalies.push("Poor market consensus");
    if (Math.abs(rho - 0.5) > 0.3) anomalies.push("Unusual experience indicators");
    
    // Trading recommendations
    const recommendations: string[] = [];
    
    if (coherence > 0.8) {
      recommendations.push("High coherence: Consider strategic positions");
    } else if (coherence < 0.3) {
      recommendations.push("Low coherence: High risk environment, avoid complex strategies");
    }
    
    if (psi > 0.8 && q < 0.3) {
      recommendations.push("Clear pricing + low emotion: Good for defined risk strategies");
    }
    
    if (q > 0.6) {
      recommendations.push("High emotion detected: Potential volatility expansion");
    }
    
    return {
      symbol,
      timestamp: new Date().toISOString(),
      gctMetrics: {
        psi: Math.round(psi * 1000) / 1000,
        rho: Math.round(rho * 1000) / 1000,
        q: Math.round(q * 1000) / 1000,
        f: Math.round(f * 1000) / 1000
      },
      coherence: Math.round(coherence * 1000) / 1000,
      assessment: coherence > 0.75 ? "HIGHLY_COHERENT" : 
                  coherence > 0.6 ? "COHERENT" :
                  coherence > 0.4 ? "MIXED" : 
                  coherence > 0.25 ? "INCOHERENT" : "HIGHLY_INCOHERENT",
      riskLevel: coherence > 0.6 ? "LOW" : coherence > 0.4 ? "MEDIUM" : "HIGH",
      anomalies,
      recommendations,
      marketMetrics: {
        putCallVolumeRatio: puts.reduce((s, p) => s + p.volume, 0) / Math.max(calls.reduce((s, c) => s + c.volume, 0), 1),
        avgImpliedVol: [...calls, ...puts].reduce((s, o) => s + o.impliedVol, 0) / (calls.length + puts.length),
        avgSpread: [...calls, ...puts].reduce((s, o) => s + (o.ask - o.bid) / ((o.ask + o.bid) / 2), 0) / (calls.length + puts.length)
      }
    };
  };

  // Perform analysis
  const performAnalysis = async () => {
    setIsAnalyzing(true);
    
    try {
      const data = sampleOptionsData[selectedStock];
      setOptionsData(data);
      
      const analysis = await analyzeOptionsCoherence(data);
      setAnalysisResults(analysis);
      
      // Add to history
      setCoherenceHistory(prev => {
        const newHistory = [...prev, {
          timestamp: new Date().toLocaleTimeString(),
          coherence: analysis.coherence,
          symbol: selectedStock,
          assessment: analysis.assessment
        }];
        return newHistory.slice(-20); // Keep last 20 points
      });
      
      // Generate alerts if needed
      if (analysis.coherence < 0.3) {
        setAlerts(prev => [...prev, {
          id: Date.now(),
          type: 'warning',
          message: `${selectedStock}: Low coherence detected (${analysis.coherence})`,
          timestamp: new Date().toLocaleTimeString()
        }].slice(-10));
      }
      
    } catch (error) {
      console.error('Analysis failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  useEffect(() => {
    performAnalysis();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedStock]);

  const getCoherenceColor = (coherence: number) => {
    if (coherence > 0.75) return '#10B981'; // Green
    if (coherence > 0.6) return '#3B82F6';  // Blue  
    if (coherence > 0.4) return '#F59E0B';  // Yellow
    if (coherence > 0.25) return '#EF4444'; // Red
    return '#7C2D12'; // Dark red
  };

  const getRiskBadgeColor = (riskLevel: string) => {
    switch(riskLevel) {
      case 'LOW': return 'bg-green-500';
      case 'MEDIUM': return 'bg-yellow-500';
      case 'HIGH': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            GCT Options Coherence Analysis
          </h1>
          <p className="text-xl text-gray-300 mb-6">
            Grounded Coherence Theory • Options Pricing Analysis • Market Sentiment Assessment
          </p>
        </div>

        {/* Controls */}
        <div className="mb-8 bg-slate-800 rounded-xl p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Target className="w-6 h-6" />
              Analysis Controls
            </h2>
            <div className="flex gap-4">
              <select 
                value={selectedStock} 
                onChange={(e) => setSelectedStock(e.target.value)}
                className="px-4 py-2 bg-slate-700 rounded-lg font-semibold"
              >
                <option value="FBK">FBK - FB Financial</option>
                <option value="AAPL">AAPL - Apple Inc</option>
                <option value="TSLA">TSLA - Tesla Inc</option>
              </select>
              <button
                onClick={performAnalysis}
                disabled={isAnalyzing}
                className="px-6 py-2 bg-blue-600 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                {isAnalyzing ? 'Analyzing...' : 'Refresh Analysis'}
              </button>
            </div>
          </div>
        </div>

        {/* Analysis Results */}
        {analysisResults && (
          <>
            {/* Main Coherence Dashboard */}
            <div className="mb-8 bg-slate-800 rounded-xl p-6">
              <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                <Brain className="w-6 h-6" />
                Coherence Assessment for {analysisResults.symbol}
              </h2>
              
              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
                {/* Overall Coherence */}
                <div className="bg-slate-700 rounded-lg p-4 text-center">
                  <div className="text-4xl font-bold mb-2" style={{color: getCoherenceColor(analysisResults.coherence)}}>
                    {(analysisResults.coherence * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-400">Overall Coherence</div>
                  <div className={`inline-block px-3 py-1 rounded-full text-xs font-bold mt-2 ${getRiskBadgeColor(analysisResults.riskLevel)}`}>
                    {analysisResults.riskLevel} RISK
                  </div>
                </div>
                
                {/* Assessment */}
                <div className="bg-slate-700 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold mb-2 text-blue-400">
                    {analysisResults.assessment.replace('_', ' ')}
                  </div>
                  <div className="text-sm text-gray-400">Market State</div>
                  <div className="mt-2">
                    {analysisResults.assessment.includes('COHERENT') ? 
                      <CheckCircle className="w-6 h-6 text-green-400 mx-auto" /> :
                      <AlertTriangle className="w-6 h-6 text-yellow-400 mx-auto" />
                    }
                  </div>
                </div>
                
                {/* P/C Ratio */}
                <div className="bg-slate-700 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold mb-2 text-purple-400">
                    {analysisResults.marketMetrics.putCallVolumeRatio.toFixed(2)}
                  </div>
                  <div className="text-sm text-gray-400">Put/Call Volume Ratio</div>
                  <div className="text-xs mt-2 text-gray-500">
                    {analysisResults.marketMetrics.putCallVolumeRatio > 1.2 ? 'Bearish Sentiment' : 
                     analysisResults.marketMetrics.putCallVolumeRatio < 0.8 ? 'Bullish Sentiment' : 'Neutral'}
                  </div>
                </div>
                
                {/* Avg IV */}
                <div className="bg-slate-700 rounded-lg p-4 text-center">
                  <div className="text-2xl font-bold mb-2 text-orange-400">
                    {(analysisResults.marketMetrics.avgImpliedVol * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-400">Avg Implied Volatility</div>
                  <div className="text-xs mt-2 text-gray-500">
                    {analysisResults.marketMetrics.avgImpliedVol > 0.4 ? 'High Volatility' :
                     analysisResults.marketMetrics.avgImpliedVol < 0.2 ? 'Low Volatility' : 'Normal'}
                  </div>
                </div>
              </div>

              {/* GCT Components Radar Chart */}
              <div className="mb-6">
                <h3 className="text-lg font-bold mb-4">GCT Component Analysis</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div className="bg-slate-700 rounded-lg p-4">
                    <ResponsiveContainer width="100%" height={300}>
                      <RadarChart data={[{
                        component: 'Clarity (ψ)',
                        value: analysisResults.gctMetrics.psi,
                        fullMark: 1
                      }, {
                        component: 'Wisdom (ρ)', 
                        value: analysisResults.gctMetrics.rho,
                        fullMark: 1
                      }, {
                        component: 'Emotion (q)',
                        value: analysisResults.gctMetrics.q,
                        fullMark: 1
                      }, {
                        component: 'Consensus (f)',
                        value: analysisResults.gctMetrics.f,
                        fullMark: 1
                      }]}> 
                        <PolarGrid />
                        <PolarAngleAxis dataKey="component" tick={{fontSize: 12}} />
                        <PolarRadiusAxis domain={[0, 1]} tick={{fontSize: 10}} />
                        <Radar dataKey="value" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                  
                  <div className="space-y-4">
                    {/* Component Breakdown */}
                    <div className="bg-slate-600 rounded-lg p-4">
                      <h4 className="font-bold mb-3">Component Breakdown</h4>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-sm">
                            <span>ψ (Pricing Clarity)</span>
                            <span className="font-bold">{(analysisResults.gctMetrics.psi * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-700 rounded-full h-2 mt-1">
                            <div 
                              className="bg-blue-400 h-2 rounded-full transition-all duration-500"
                              style={{width: `${analysisResults.gctMetrics.psi * 100}%`}}
                            ></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-sm">
                            <span>ρ (Market Wisdom)</span>
                            <span className="font-bold">{(analysisResults.gctMetrics.rho * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-700 rounded-full h-2 mt-1">
                            <div 
                              className="bg-green-400 h-2 rounded-full transition-all duration-500"
                              style={{width: `${analysisResults.gctMetrics.rho * 100}%`}}
                            ></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-sm">
                            <span>q (Emotional Charge)</span>
                            <span className="font-bold">{(analysisResults.gctMetrics.q * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-700 rounded-full h-2 mt-1">
                            <div 
                              className="bg-red-400 h-2 rounded-full transition-all duration-500"
                              style={{width: `${analysisResults.gctMetrics.q * 100}%`}}
                            ></div>
                          </div>
                        </div>
                        <div>
                          <div className="flex justify-between text-sm">
                            <span>f (Market Consensus)</span>
                            <span className="font-bold">{(analysisResults.gctMetrics.f * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-slate-700 rounded-full h-2 mt-1">
                            <div 
                              className="bg-purple-400 h-2 rounded-full transition-all duration-500"
                              style={{width: `${analysisResults.gctMetrics.f * 100}%`}}
                            ></div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Alerts and Recommendations */}
              {(analysisResults.anomalies.length > 0 || analysisResults.recommendations.length > 0) && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Anomalies */}
                  {analysisResults.anomalies.length > 0 && (
                    <div className="bg-red-900 bg-opacity-30 border border-red-600 rounded-lg p-4">
                      <h4 className="font-bold text-red-400 mb-3 flex items-center gap-2">
                        <AlertTriangle className="w-5 h-5" />
                        Detected Anomalies
                      </h4>
                      <ul className="space-y-2">
                        {analysisResults.anomalies.map((anomaly: string, idx: number) => (
                          <li key={idx} className="text-sm text-red-300">• {anomaly}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                  
                  {/* Recommendations */}
                  {analysisResults.recommendations.length > 0 && (
                    <div className="bg-green-900 bg-opacity-30 border border-green-600 rounded-lg p-4">
                      <h4 className="font-bold text-green-400 mb-3 flex items-center gap-2">
                        <Zap className="w-5 h-5" />
                        Trading Insights
                      </h4>
                      <ul className="space-y-2">
                        {analysisResults.recommendations.map((rec: string, idx: number) => (
                          <li key={idx} className="text-sm text-green-300">• {rec}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Coherence History */}
            {coherenceHistory.length > 0 && (
              <div className="mb-8 bg-slate-800 rounded-xl p-6">
                <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <Activity className="w-6 h-6" />
                  Coherence Timeline
                </h2>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={coherenceHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis dataKey="timestamp" stroke="#9CA3AF" />
                    <YAxis domain={[0, 1]} stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: '#1F2937',
                        border: '1px solid #374151',
                        borderRadius: '8px'
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="coherence" 
                      stroke="#8B5CF6" 
                      strokeWidth={3}
                      dot={{ fill: '#8B5CF6', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Options Chain Visualization */}
            {optionsData && (
              <div className="mb-8 bg-slate-800 rounded-xl p-6">
                <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <BarChart3 className="w-6 h-6" />
                  Options Chain Analysis
                </h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Calls */}
                  <div>
                    <h3 className="text-lg font-bold mb-4 text-green-400">Call Options</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-slate-600">
                            <th className="text-left py-2">Strike</th>
                            <th className="text-left py-2">Bid/Ask</th>
                            <th className="text-left py-2">IV</th>
                            <th className="text-left py-2">Volume</th>
                          </tr>
                        </thead>
                        <tbody>
                          {optionsData.calls.map((call: any, idx: number) => (
                            <tr key={idx} className="border-b border-slate-700">
                              <td className="py-2 font-bold">${call.strike}</td>
                              <td className="py-2">${call.bid.toFixed(2)} / ${call.ask.toFixed(2)}</td>
                              <td className="py-2">{(call.impliedVol * 100).toFixed(1)}%</td>
                              <td className="py-2">{call.volume}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Puts */}
                  <div>
                    <h3 className="text-lg font-bold mb-4 text-red-400">Put Options</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-slate-600">
                            <th className="text-left py-2">Strike</th>
                            <th className="text-left py-2">Bid/Ask</th>
                            <th className="text-left py-2">IV</th>
                            <th className="text-left py-2">Volume</th>
                          </tr>
                        </thead>
                        <tbody>
                          {optionsData.puts.map((put: any, idx: number) => (
                            <tr key={idx} className="border-b border-slate-700">
                              <td className="py-2 font-bold">${put.strike}</td>
                              <td className="py-2">${put.bid.toFixed(2)} / ${put.ask.toFixed(2)}</td>
                              <td className="py-2">{(put.impliedVol * 100).toFixed(1)}%</td>
                              <td className="py-2">{put.volume}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Alerts */}
            {alerts.length > 0 && (
              <div className="bg-slate-800 rounded-xl p-6">
                <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
                  <Eye className="w-6 h-6" />
                  Recent Alerts
                </h2>
                <div className="space-y-3">
                  {alerts.map((alert) => (
                    <div key={alert.id} className="bg-yellow-900 bg-opacity-30 border border-yellow-600 rounded-lg p-4">
                      <div className="flex justify-between items-center">
                        <span className="text-yellow-300">{alert.message}</span>
                        <span className="text-xs text-yellow-500">{alert.timestamp}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}

        {/* Footer */}
        <div className="mt-12 text-center text-gray-400">
          <p className="text-sm">
            GCT Options Analysis: Using Grounded Coherence Theory to assess options market structure and sentiment
          </p>
        </div>
      </div>
    </div>
  );
};

export default OptionsCoherenceAnalyzer;
