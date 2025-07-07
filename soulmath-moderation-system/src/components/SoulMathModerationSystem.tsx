import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
         BarChart, Bar, ScatterChart, Scatter, Cell } from 'recharts';
import { Shield, AlertTriangle, CheckCircle, XCircle, Eye, Bot, Users, TrendingDown, Zap, Brain } from 'lucide-react';

interface ModerationAnalysis {
  psi: number;
  rho: number;
  coherence: number;
  toxicityRisk: number;
  manipulationRisk: number;
  extremismRisk: number;
  spamRisk: number;
  harassmentRisk: number;
  discourseCollapse: number;
  escalationRisk: number;
  behaviorDrift: number;
  overallRisk: number;
  textLength: number;
  sentenceCount: number;
  witnessMultiplier: number;
}

interface ModerationAction {
  type: 'remove' | 'flag' | 'review' | 'approve';
  severity: 'high' | 'medium' | 'low' | 'none';
  reason: string;
}

interface ContentItem {
  id: number;
  text: string;
  user: string;
  timestamp: string;
  context: string;
  analysis?: ModerationAnalysis;
  action?: ModerationAction;
  confidence?: number;
}

const SoulMathModerationSystem: React.FC = () => {
  const [moderationQueue, setModerationQueue] = useState<ContentItem[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeFilters, setActiveFilters] = useState({
    harassment: true,
    manipulation: true,
    toxicity: true,
    spam: true,
    extremism: true
  });
  const [systemStats, setSystemStats] = useState({
    processed: 2891,
    flagged: 187,
    escalated: 23,
    accuracy: 91.7
  });

  // Enhanced SoulMath Moderation Engine
  const analyzeModerationRisk = (text: string, userHistory: any[] = []): ModerationAnalysis => {
    const words = text.toLowerCase().split(/\s+/);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    
    // Core coherence metrics
    const avgWordsPerSentence = words.length / Math.max(sentences.length, 1);
    const psi = Math.min(1.0, 0.3 + (avgWordsPerSentence / 20));
    const uniqueWords = new Set(words).size;
    const rho = Math.min(1.0, uniqueWords / Math.max(words.length, 1) + 0.2);
    
    // Toxicity indicators (enhanced emotional charge)
    const toxicPatterns = [
      'hate', 'kill', 'die', 'stupid', 'idiot', 'loser', 'waste', 'trash', 'scum',
      'moron', 'worthless', 'disgusting', 'retard', 'gay', 'pathetic', 'fag'
    ];
    
    const harassmentPatterns = [
      'you should', 'why don\'t you', 'go back to', 'your kind', 'typical',
      'always complaining', 'people like you', 'cry more'
    ];
    
    const toxicScore = words.filter(w => toxicPatterns.includes(w)).length;
    const harassmentScore = harassmentPatterns.filter(p => 
      text.toLowerCase().includes(p)).length;
    const q_toxic = Math.min(1.0, (toxicScore + harassmentScore) / Math.max(words.length, 1) * 20);
    
    // Manipulation detection (symbolic recurrence)
    const manipulationPatterns = [
      'everyone knows', 'obviously', 'wake up', 'sheep', 'brainwashed',
      'any reasonable person', 'it\'s clear that', 'open your eyes', 'the truth is'
    ];
    
    const gaslightingPatterns = [
      'you\'re imagining', 'that never happened', 'you\'re too sensitive',
      'you misunderstood', 'calm down', 'you\'re overreacting'
    ];
    
    const manipulationScore = manipulationPatterns.filter(p => 
      text.toLowerCase().includes(p)).length;
    const gaslightingScore = gaslightingPatterns.filter(p => 
      text.toLowerCase().includes(p)).length;
    const f_manipulation = Math.min(1.0, (manipulationScore + gaslightingScore) / 10);
    
    // Extremism detection
    const extremismPatterns = [
      'all [group] are', 'destroy the', 'war on', 'invasion', 'replacement',
      'eliminate', 'purge', 'they want to', 'cleanse', 'taking over'
    ];
    
    const extremismScore = extremismPatterns.filter(p => 
      new RegExp(p.replace('[group]', '\\\\w+')).test(text.toLowerCase())
    ).length;
    
    // Spam/bot detection
    const repetitivePatterns = words.reduce((acc: Record<string, number>, word) => {
      acc[word] = (acc[word] || 0) + 1;
      return acc;
    }, {});
    
    const maxRepetition = Math.max(...Object.values(repetitivePatterns));
    const spamScore = maxRepetition > words.length * 0.3 ? 1 : 0;
    
    // Advanced moderation metrics
    const toxicityRisk = q_toxic;
    const manipulationRisk = f_manipulation;
    const extremismRisk = Math.min(1.0, extremismScore / 5);
    const spamRisk = spamScore;
    const harassmentRisk = Math.min(1.0, harassmentScore / 3);
    
    // Coherence collapse in moderation context = loss of rational discourse
    const discourseCollapse = Math.max(0, 1 - (psi * (1 - toxicityRisk) * rho) / 3);
    
    // Escalation risk - likelihood of causing community harm
    const escalationRisk = (toxicityRisk + manipulationRisk + extremismRisk + harassmentRisk) / 4;
    
    // User pattern analysis (simplified)
    const userCoherenceTrend = userHistory.length > 0 ?
      userHistory.reduce((sum, post) => sum + (post.coherence || 0.5), 0) / userHistory.length :
      0.5;
    const behaviorDrift = Math.abs(psi - userCoherenceTrend);
    
    // Witness integration for moderation context
    const communityWitnessQuality = 0.8; // Community standards adherence
    const moderationDepth = 0.7; // How deeply to analyze
    const witnessMultiplier = 1 + (0.1 * moderationDepth) * communityWitnessQuality;
    
    // Final risk assessment
    const overallRisk = (escalationRisk * witnessMultiplier);
    
    return {
      // Core metrics
      psi: parseFloat(psi.toFixed(3)),
      rho: parseFloat(rho.toFixed(3)),
      coherence: parseFloat(psi.toFixed(3)),
      
      // Moderation-specific risks
      toxicityRisk: parseFloat(toxicityRisk.toFixed(3)),
      manipulationRisk: parseFloat(manipulationRisk.toFixed(3)),
      extremismRisk: parseFloat(extremismRisk.toFixed(3)),
      spamRisk: parseFloat(spamRisk.toFixed(3)),
      harassmentRisk: parseFloat(harassmentRisk.toFixed(3)),
      
      // Advanced metrics
      discourseCollapse: parseFloat(discourseCollapse.toFixed(3)),
      escalationRisk: parseFloat(escalationRisk.toFixed(3)),
      behaviorDrift: parseFloat(behaviorDrift.toFixed(3)),
      overallRisk: parseFloat(overallRisk.toFixed(3)),
      
      // Metadata
      textLength: words.length,
      sentenceCount: sentences.length,
      witnessMultiplier: parseFloat(witnessMultiplier.toFixed(3))
    };
  };

  // Sample content for demonstration based on Reddit research
  const sampleContent: ContentItem[] = [
    {
      id: 1,
      text: "I really disagree with this policy. I think we need to have a more nuanced discussion about the implications and consider alternative approaches.",
      user: "thoughtful_user",
      timestamp: "2 min ago",
      context: "Political discussion"
    },
    {
      id: 2,
      text: "These people are absolutely disgusting and shouldn't be allowed in our country. They're destroying everything we've built and someone needs to stop them.",
      user: "angry_poster",
      timestamp: "5 min ago",
      context: "Immigration thread"
    },
    {
      id: 3,
      text: "You're clearly too stupid to understand basic economics. Maybe go back to school before embarrassing yourself with these moronic takes.",
      user: "hostile_debater",
      timestamp: "8 min ago",
      context: "Economic policy"
    },
    {
      id: 4,
      text: "AMAZING OPPORTUNITY!!! Make $5000 a week from home! Click here now! Limited time offer! Don't miss out! Financial freedom awaits!",
      user: "definitely_not_spam",
      timestamp: "12 min ago",
      context: "Career advice"
    },
    {
      id: 5,
      text: "You're imagining things again. That never happened and you know it. You're being way too sensitive about this - everyone else understands the joke.",
      user: "gaslighter_2024",
      timestamp: "15 min ago",
      context: "Community complaint"
    },
    {
      id: 6,
      text: "Are you fucking retarded? Do you know how stupid you sound? Of course, it's to be expected from you. People like you always complaining about everything.",
      user: "repeat_harasser",
      timestamp: "18 min ago",
      context: "r/politics"
    },
    {
      id: 7,
      text: "I wasn't actually reading those books and you cherry-picked those to be over-the-top. You people are all the same, trying to act smarter than everyone else.",
      user: "book_critic",
      timestamp: "22 min ago",
      context: "r/books AMA"
    },
    {
      id: 8,
      text: "Black scientists getting downvoted again? Typical. Wake up sheep, this is obviously coordinated by the establishment to silence real voices.",
      user: "conspiracy_theorist",
      timestamp: "25 min ago",
      context: "r/science"
    },
    {
      id: 9,
      text: "Thanks for sharing this research. While I have some questions about the methodology, I appreciate the effort that went into this study.",
      user: "constructive_critic",
      timestamp: "28 min ago",
      context: "r/science"
    },
    {
      id: 10,
      text: "Your kind always does this. Go back to where you came from. We don't need more diversity hires ruining our communities.",
      user: "identity_attacker",
      timestamp: "30 min ago",
      context: "Diversity discussion"
    }
  ];

  const processContent = () => {
    setIsProcessing(true);
    setTimeout(() => {
      const analyzedContent = sampleContent.map(item => {
        const analysis = analyzeModerationRisk(item.text);
        const action = determineAction(analysis);
        return {
          ...item,
          analysis,
          action,
          confidence: Math.random() * 0.3 + 0.7 // 70-100% confidence
        };
      });
      setModerationQueue(analyzedContent);
      setIsProcessing(false);
    }, 2000);
  };

  const determineAction = (analysis: ModerationAnalysis): ModerationAction => {
    const { overallRisk, toxicityRisk, extremismRisk, spamRisk, discourseCollapse } = analysis;
    
    if (extremismRisk > 0.6 || overallRisk > 0.8) {
      return { type: 'remove', severity: 'high', reason: 'Extremism/High toxicity' };
    } else if (toxicityRisk > 0.5 || discourseCollapse > 0.7) {
      return { type: 'flag', severity: 'medium', reason: 'Toxic behavior' };
    } else if (spamRisk > 0.8) {
      return { type: 'remove', severity: 'medium', reason: 'Spam detection' };
    } else if (overallRisk > 0.4) {
      return { type: 'review', severity: 'low', reason: 'Potential issue' };
    } else {
      return { type: 'approve', severity: 'none', reason: 'Content acceptable' };
    }
  };

  const getActionColor = (action: ModerationAction) => {
    switch (action.type) {
      case 'remove': return action.severity === 'high' ? 'red' : 'orange';
      case 'flag': return 'yellow';
      case 'review': return 'blue';
      case 'approve': return 'green';
      default: return 'gray';
    }
  };

  const getActionIcon = (action: ModerationAction) => {
    switch (action.type) {
      case 'remove': return <XCircle className="w-5 h-5" />;
      case 'flag': return <AlertTriangle className="w-5 h-5" />;
      case 'review': return <Eye className="w-5 h-5" />;
      case 'approve': return <CheckCircle className="w-5 h-5" />;
      default: return <Shield className="w-5 h-5" />;
    }
  };

  useEffect(() => {
    // Auto-process on mount
    processContent();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-slate-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            SoulMath AI Moderation System
          </h1>
          <p className="text-xl text-gray-300 mb-6">
            Coherence-based content analysis • Early intervention • Community protection
          </p>
          <div className="flex justify-center gap-6 text-sm">
            <div className="bg-slate-800 px-4 py-2 rounded-lg">
              <span className="text-gray-400">Processed: </span>
              <span className="text-green-400 font-bold">{systemStats.processed}</span>
            </div>
            <div className="bg-slate-800 px-4 py-2 rounded-lg">
              <span className="text-gray-400">Flagged: </span>
              <span className="text-yellow-400 font-bold">{systemStats.flagged}</span>
            </div>
            <div className="bg-slate-800 px-4 py-2 rounded-lg">
              <span className="text-gray-400">Accuracy: </span>
              <span className="text-blue-400 font-bold">{systemStats.accuracy}%</span>
            </div>
          </div>
        </div>

        {/* Controls */}
        <div className="mb-8 bg-slate-800 rounded-xl p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Bot className="w-6 h-6" />
              Moderation Controls
            </h2>
            <div className="flex gap-4">
              <button
                onClick={processContent}
                disabled={isProcessing}
                className="px-6 py-2 bg-blue-600 rounded-lg font-semibold hover:bg-blue-700 disabled:opacity-50 transition-colors"
              >
                {isProcessing ? 'Processing...' : 'Reprocess Queue'}
              </button>
              <button
                onClick={() => {
                  setSystemStats(prev => ({
                    ...prev,
                    processed: prev.processed + 47,
                    flagged: prev.flagged + 8,
                    escalated: prev.escalated + 2
                  }));
                }}
                className="px-4 py-2 bg-purple-600 rounded-lg font-semibold hover:bg-purple-700 transition-colors"
              >
                Simulate Live Feed
              </button>
            </div>
          </div>
          
          <div className="flex flex-wrap gap-4">
            {Object.entries(activeFilters).map(([filter, active]) => (
              <button
                key={filter}
                onClick={() => setActiveFilters(prev => ({ ...prev, [filter]: !active }))}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  active
                    ? 'bg-purple-600 text-white'
                    : 'bg-slate-700 text-gray-400 hover:bg-slate-600'
                }`}
              >
                {filter.charAt(0).toUpperCase() + filter.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Analysis Summary */}
        {moderationQueue.length > 0 && (
          <div className="mb-8 bg-slate-800 rounded-xl p-6">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
              <TrendingDown className="w-6 h-6" />
              SoulMath Analysis Summary
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-slate-700 rounded-lg p-4">
                <h3 className="font-bold text-green-400 mb-2">Healthy Discourse</h3>
                <p className="text-sm text-gray-300 mb-2">
                  Comments with high coherence (ψ &gt; 0.7) and low toxicity risk
                </p>
                <div className="text-2xl font-bold">
                  {moderationQueue.filter(item => 
                    item.analysis && item.analysis.coherence > 0.7 && item.analysis.toxicityRisk < 0.3
                  ).length}
                </div>
              </div>
              
              <div className="bg-slate-700 rounded-lg p-4">
                <h3 className="font-bold text-yellow-400 mb-2">Discourse Collapse Risk</h3>
                <p className="text-sm text-gray-300 mb-2">
                  Comments showing breakdown of rational discussion
                </p>
                <div className="text-2xl font-bold">
                  {moderationQueue.filter(item => 
                    item.analysis && item.analysis.discourseCollapse > 0.5
                  ).length}
                </div>
              </div>
              
              <div className="bg-slate-700 rounded-lg p-4">
                <h3 className="font-bold text-red-400 mb-2">High Risk Accounts</h3>
                <p className="text-sm text-gray-300 mb-2">
                  Comments requiring immediate intervention
                </p>
                <div className="text-2xl font-bold">
                  {moderationQueue.filter(item => 
                    item.analysis && item.analysis.overallRisk > 0.7
                  ).length}
                </div>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-slate-600 rounded-lg">
              <h4 className="font-bold text-blue-400 mb-2">SoulMath Insights</h4>
              <p className="text-sm text-gray-300">
                The coherence engine detected {moderationQueue.filter(item => 
                  item.analysis && item.analysis.behaviorDrift > 0.4
                ).length} users showing significant behavioral drift, suggesting potential influence from toxic community dynamics.
                Average discourse collapse index: {(moderationQueue.reduce((sum, item) => 
                  sum + (item.analysis?.discourseCollapse || 0), 0
                ) / moderationQueue.length).toFixed(3)}
              </p>
            </div>
          </div>
        )}

        {/* Risk Dashboard */}
        {moderationQueue.length > 0 && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div className="bg-slate-800 rounded-xl p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Zap className="w-6 h-6" />
                Risk Analysis Overview
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <ScatterChart data={moderationQueue.filter(item => item.analysis)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis
                    type="number"
                    dataKey="analysis.overallRisk"
                    domain={[0, 1]}
                    stroke="#9CA3AF"
                    name="Overall Risk"
                  />
                  <YAxis
                    type="number"
                    dataKey="analysis.discourseCollapse"
                    domain={[0, 1]}
                    stroke="#9CA3AF"
                    name="Discourse Collapse"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                    formatter={(value: any, name: any) => [
                      typeof value === 'number' ? value.toFixed(3) : value,
                      name
                    ]}
                  />
                  <Scatter dataKey="analysis.escalationRisk" fill="#8884d8">
                    {moderationQueue.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={
                          entry.action?.type === 'remove' ? '#EF4444' :
                          entry.action?.type === 'flag' ? '#F59E0B' :
                          entry.action?.type === 'review' ? '#3B82F6' :
                          '#10B981'
                        }
                      />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
            
            <div className="bg-slate-800 rounded-xl p-6">
              <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
                <Brain className="w-6 h-6" />
                Risk Category Breakdown
              </h2>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={moderationQueue.filter(item => item.analysis)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="id" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={[0, 1]} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="analysis.toxicityRisk" fill="#EF4444" name="Toxicity" />
                  <Bar dataKey="analysis.manipulationRisk" fill="#F59E0B" name="Manipulation" />
                  <Bar dataKey="analysis.extremismRisk" fill="#DC2626" name="Extremism" />
                  <Bar dataKey="analysis.spamRisk" fill="#6B7280" name="Spam" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Moderation Queue */}
        <div className="bg-slate-800 rounded-xl p-6">
          <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
            <Users className="w-6 h-6" />
            Content Moderation Queue
          </h2>
          <div className="space-y-4">
            {moderationQueue.map((item) => (
              <div
                key={item.id}
                className={`bg-slate-700 rounded-lg p-5 border-l-4 ${
                  item.action?.type === 'remove' ? 'border-red-500' :
                  item.action?.type === 'flag' ? 'border-yellow-500' :
                  item.action?.type === 'review' ? 'border-blue-500' :
                  'border-green-500'
                }`}
              >
                <div className="flex justify-between items-start mb-3">
                  <div className="flex items-center gap-3">
                    <div className="p-2 rounded-full bg-slate-600">
                      {item.action && getActionIcon(item.action)}
                    </div>
                    <div>
                      <div className="font-semibold text-lg">
                        {item.action?.type.toUpperCase()} - {item.action?.reason}
                      </div>
                      <div className="text-sm text-gray-400">
                        @{item.user} • {item.timestamp} • {item.context}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-400">Confidence</div>
                    <div className="font-bold text-lg">
                      {item.confidence ? `${(item.confidence * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>
                </div>
                
                <div className="bg-slate-600 rounded p-3 mb-4">
                  <p className="text-gray-200 italic">"{item.text}"</p>
                </div>
                
                {item.analysis && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-400">Overall Risk:</span>
                      <div className={`font-bold ${
                        item.analysis.overallRisk > 0.7 ? 'text-red-400' :
                        item.analysis.overallRisk > 0.4 ? 'text-yellow-400' :
                        'text-green-400'
                      }`}>
                        {(item.analysis.overallRisk * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-400">Toxicity:</span>
                      <div className="font-bold text-red-400">
                        {(item.analysis.toxicityRisk * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-400">Discourse Collapse:</span>
                      <div className="font-bold text-orange-400">
                        {(item.analysis.discourseCollapse * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-400">Coherence:</span>
                      <div className="font-bold text-blue-400">
                        {(item.analysis.coherence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                )}
                
                <div className="flex gap-3 mt-4">
                  <button className="px-4 py-2 bg-green-600 rounded hover:bg-green-700 transition-colors">
                    Approve
                  </button>
                  <button className="px-4 py-2 bg-red-600 rounded hover:bg-red-700 transition-colors">
                    Remove
                  </button>
                  <button className="px-4 py-2 bg-yellow-600 rounded hover:bg-yellow-700 transition-colors">
                    Flag User
                  </button>
                  <button className="px-4 py-2 bg-purple-600 rounded hover:bg-purple-700 transition-colors">
                    Escalate
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="mt-12 text-center text-gray-400">
          <p className="text-sm">
            SoulMath Moderation: Using coherence mathematics to protect community discourse 
            and prevent toxicity escalation
          </p>
        </div>
      </div>
    </div>
  );
};

export default SoulMathModerationSystem;