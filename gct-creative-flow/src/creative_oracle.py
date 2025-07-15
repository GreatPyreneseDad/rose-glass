"""
The Creative Oracle - Mystical guidance for your creative journey
Using GCT patterns to provide personalized creative wisdom
"""

import random
from typing import Dict, List, Tuple
from datetime import datetime
import numpy as np

class CreativeOracle:
    """A mystical guide that interprets your creative energy"""
    
    def __init__(self):
        # Oracle wisdom database
        self.flow_mantras = [
            "🌊 You are the river and the riverbed. Flow without forcing.",
            "🎯 Your clarity pierces through - trust this vision completely.",
            "✨ This is your golden hour. Create without questioning.",
            "🔥 The muse dances with you now. Lead with confidence.",
            "💫 You've found the sweet spot between effort and ease."
        ]
        
        self.exploration_guidance = [
            "🗺️ Not all who wander are lost - your exploration serves a purpose.",
            "🌱 Plant seeds of curiosity everywhere. Some will surprise you.",
            "🔍 The answer hides in the question you haven't asked yet.",
            "🎪 Play is the highest form of research. Be silly.",
            "🌈 Follow the weird idea - it knows something you don't."
        ]
        
        self.blocked_remedies = [
            "🚶 Walk. The feet move, the mind follows, the block dissolves.",
            "💧 Water your other gardens. Return when you're ready.",
            "🎭 Do the opposite of what feels 'right'. Surprise yourself.",
            "📝 Write badly on purpose. Make it terrible. Then laugh.",
            "🌙 The block is a teacher in disguise. What's the lesson?"
        ]
        
        self.incubation_whispers = [
            "🌙 Trust the darkness. Stars need night to shine.",
            "🍃 Let it simmer. The best ideas cook slowly.",
            "☁️ Daydream without guilt. You're working in the space between.",
            "🦋 The cocoon phase looks like nothing. Everything is happening.",
            "💤 Rest is a creative act. Your subconscious is busy."
        ]
        
        self.illumination_celebrations = [
            "💡 EUREKA! Capture this lightning in every bottle you can find!",
            "🎆 The universe just winked at you. Wink back with your work.",
            "⚡ This is why you create. Remember this feeling.",
            "🏔️ You've reached a peak. Look how far you've come!",
            "🌟 Your inner genius just said hello. Have a conversation."
        ]
        
        # Creative archetypes
        self.archetypes = {
            "The Alchemist": {
                "description": "You transform raw experience into creative gold",
                "strength": "Seeing connections others miss",
                "challenge": "Perfectionism blocks your flow",
                "totem": "🔮"
            },
            "The Wild Child": {
                "description": "Your creativity is pure play and wonder",
                "strength": "Fearless experimentation",
                "challenge": "Finishing what you start",
                "totem": "🎨"
            },
            "The Deep Diver": {
                "description": "You plunge into the depths for pearls of wisdom",
                "strength": "Profound insights and meaning",
                "challenge": "Coming up for air",
                "totem": "🌊"
            },
            "The Lightning Rod": {
                "description": "You channel creative energy in powerful bursts",
                "strength": "Intense focus and breakthrough moments",
                "challenge": "Sustaining energy between strikes",
                "totem": "⚡"
            },
            "The Garden Keeper": {
                "description": "You cultivate ideas with patience and care",
                "strength": "Nurturing projects to full bloom",
                "challenge": "Letting go when it's time",
                "totem": "🌱"
            }
        }
        
        # Cosmic creative cycles
        self.cosmic_phases = [
            "New Moon (Beginning)",
            "Waxing Crescent (Building)",
            "First Quarter (Deciding)", 
            "Waxing Gibbous (Refining)",
            "Full Moon (Illumination)",
            "Waning Gibbous (Sharing)",
            "Last Quarter (Releasing)",
            "Waning Crescent (Resting)"
        ]
        
        # Creative medicine
        self.creative_medicine = {
            "low_clarity": [
                "🧹 Clear your workspace, clear your mind",
                "📋 Make a tiny list. Do one thing.",
                "🎵 Change your sonic environment",
                "💨 Breathe like you mean it"
            ],
            "low_energy": [
                "🍎 Feed your body, fuel your mind",
                "💃 Move your body in a new way",
                "☀️ Find some sunlight or bright light",
                "🎪 Do something purely for fun"
            ],
            "low_connection": [
                "📞 Share your work with someone who gets it",
                "👥 Co-create, even for 10 minutes",
                "💌 Write a letter to your future creative self",
                "🌍 Remember why your work matters"
            ],
            "low_depth": [
                "📚 Read something that stretches your mind",
                "🏛️ Visit a museum, even virtually",
                "❓ Ask yourself the hard questions",
                "🕰️ Give yourself permission to go slow"
            ]
        }
    
    def divine_message(self, coherence: float, state: str, 
                      components: Dict[str, float]) -> Dict[str, str]:
        """Divine a personalized message based on current state"""
        
        # Select primary message based on state
        if state == "Flow":
            primary = random.choice(self.flow_mantras)
        elif state == "Exploration":
            primary = random.choice(self.exploration_guidance)
        elif state == "Blocked":
            primary = random.choice(self.blocked_remedies)
        elif state == "Incubation":
            primary = random.choice(self.incubation_whispers)
        elif state == "Illumination":
            primary = random.choice(self.illumination_celebrations)
        else:
            primary = "🌀 You're in transition. Trust the process."
        
        # Determine archetype
        archetype = self._divine_archetype(components)
        
        # Get specific medicine if needed
        medicine = self._prescribe_medicine(components)
        
        # Determine cosmic phase
        cosmic_phase = self._calculate_cosmic_phase()
        
        return {
            "primary_message": primary,
            "archetype": archetype,
            "medicine": medicine,
            "cosmic_phase": cosmic_phase,
            "oracle_says": self._generate_oracle_saying(coherence, state)
        }
    
    def _divine_archetype(self, components: Dict[str, float]) -> Dict:
        """Determine creative archetype based on component profile"""
        
        psi = components.get('psi', 0.5)
        rho = components.get('rho', 0.5)
        q = components.get('q', 0.5)
        f = components.get('f', 0.5)
        
        # Simple archetype matching
        if rho > 0.7 and psi > 0.6:
            return self.archetypes["The Alchemist"]
        elif q > 0.7 and f > 0.6:
            return self.archetypes["The Wild Child"]
        elif rho > 0.7 and q < 0.4:
            return self.archetypes["The Deep Diver"]
        elif psi > 0.8 and abs(q - 0.5) < 0.2:
            return self.archetypes["The Lightning Rod"]
        else:
            return self.archetypes["The Garden Keeper"]
    
    def _prescribe_medicine(self, components: Dict[str, float]) -> List[str]:
        """Prescribe creative medicine for low components"""
        
        medicine = []
        
        if components.get('psi', 1) < 0.4:
            medicine.extend(random.sample(self.creative_medicine['low_clarity'], 2))
        if components.get('q', 1) < 0.3:
            medicine.extend(random.sample(self.creative_medicine['low_energy'], 2))
        if components.get('f', 1) < 0.3:
            medicine.append(random.choice(self.creative_medicine['low_connection']))
        if components.get('rho', 1) < 0.4:
            medicine.append(random.choice(self.creative_medicine['low_depth']))
        
        return medicine[:3]  # Maximum 3 prescriptions
    
    def _calculate_cosmic_phase(self) -> str:
        """Calculate current cosmic creative phase"""
        
        # Simple cycling based on day of month
        day = datetime.now().day
        phase_index = (day - 1) % len(self.cosmic_phases)
        return self.cosmic_phases[phase_index]
    
    def _generate_oracle_saying(self, coherence: float, state: str) -> str:
        """Generate a mystical oracle saying"""
        
        sayings = {
            "high_coherence": [
                "The creative cosmos aligns with your intention",
                "You have become a clear channel for inspiration",
                "The muse recognizes you as her favorite",
                "Your creative fire burns true and bright"
            ],
            "medium_coherence": [
                "The path reveals itself step by step",
                "Creative waves ebb and flow - ride them wisely",
                "You dance between order and chaos",
                "The work teaches you as you create"
            ],
            "low_coherence": [
                "Even diamonds need pressure to form",
                "The creative soul needs both sun and storm",
                "Confusion is the beginning of wisdom",
                "Trust the mess - it's part of the magic"
            ]
        }
        
        if coherence > 0.7:
            category = "high_coherence"
        elif coherence > 0.4:
            category = "medium_coherence"
        else:
            category = "low_coherence"
        
        return random.choice(sayings[category])
    
    def generate_creative_ritual(self, state: str, time_available: int = 10) -> Dict:
        """Generate a personalized creative ritual"""
        
        rituals = {
            "Flow": {
                "name": "Flow Keeper Ritual",
                "steps": [
                    "Light a candle or turn on your 'flow light'",
                    "Write your intention in one sentence",
                    "Set a gentle timer for deep work",
                    "Begin with your breath, end with gratitude"
                ]
            },
            "Blocked": {
                "name": "Block Breaker Ritual",
                "steps": [
                    "Stand up and shake your whole body for 30 seconds",
                    "Write 3 terrible ideas on purpose",
                    "Draw or doodle for 2 minutes",
                    "Ask yourself: 'What would easy look like?'"
                ]
            },
            "Exploration": {
                "name": "Explorer's Ritual",
                "steps": [
                    "Pull 3 random books and read one paragraph from each",
                    "Ask 'What if...' 5 times and write the questions",
                    "Take 5 photos of interesting textures or shapes",
                    "Connect two unrelated things in your work"
                ]
            },
            "Incubation": {
                "name": "Dreamer's Ritual",
                "steps": [
                    "Lie down with your notebook nearby",
                    "Breathe deeply and let your mind wander for 5 minutes",
                    "Capture any images or feelings that arise",
                    "Trust that the work continues beneath awareness"
                ]
            }
        }
        
        if state in rituals:
            ritual = rituals[state]
        else:
            ritual = {
                "name": "Transition Ritual",
                "steps": [
                    "Acknowledge where you are without judgment",
                    "Choose one small creative act",
                    "Do it with full presence",
                    "Notice what wants to emerge next"
                ]
            }
        
        # Adjust for time
        if time_available < 10:
            ritual["steps"] = ritual["steps"][:2]
            ritual["duration"] = "5 minutes"
        elif time_available < 20:
            ritual["duration"] = "10-15 minutes"
        else:
            ritual["duration"] = "20-30 minutes"
            ritual["steps"].append("Journal about what emerged")
        
        return ritual
    
    def cast_creative_forecast(self, history: List[Dict]) -> Dict:
        """Forecast creative weather based on patterns"""
        
        if len(history) < 3:
            return {
                "forecast": "☁️ Still gathering cosmic data...",
                "advice": "Keep tracking to unlock your patterns"
            }
        
        # Analyze recent trends
        recent_coherences = [h.get('coherence', 0.5) for h in history[-10:]]
        recent_states = [h.get('state', 'Unknown') for h in history[-10:]]
        
        avg_coherence = np.mean(recent_coherences)
        coherence_trend = recent_coherences[-1] - recent_coherences[0]
        
        # Count state frequencies
        flow_count = recent_states.count('Flow')
        blocked_count = recent_states.count('Blocked')
        
        # Generate forecast
        if coherence_trend > 0.1:
            forecast = "☀️ Creative sunshine ahead! Your energy is rising."
            advice = "Prepare for breakthrough moments. Have capture tools ready."
        elif coherence_trend < -0.1:
            forecast = "🌧️ Creative storms brewing. Time for gentle self-care."
            advice = "Don't push. Let the rain water new seeds."
        elif flow_count > len(recent_states) / 3:
            forecast = "🌈 You're in a golden period. Harvest this energy!"
            advice = "Document your process. You'll want to recreate this."
        elif blocked_count > len(recent_states) / 3:
            forecast = "🌪️ Transformative turbulence. Change is coming."
            advice = "The block is about to teach you something important."
        else:
            forecast = "🌤️ Variable creative weather. Perfect for experiments."
            advice = "Try different approaches. See what the day brings."
        
        return {
            "forecast": forecast,
            "advice": advice,
            "cosmic_tip": self._generate_cosmic_tip()
        }
    
    def _generate_cosmic_tip(self) -> str:
        """Generate a cosmic creative tip"""
        
        tips = [
            "🌟 Create at the same time each day to build cosmic rhythm",
            "🔮 Keep a dream journal - night creativity counts too",
            "🎭 Change your creative persona - wear a different hat",
            "🗝️ The door to creativity opens from the inside",
            "🦋 Small actions create big transformations",
            "🌸 Your creativity blooms in its own season",
            "⚖️ Balance structure and chaos for optimal flow",
            "🎪 Make your workspace a sacred creative circus",
            "💎 Pressure creates diamonds and breakthroughs",
            "🌀 Spiral thinking beats straight lines every time"
        ]
        
        return random.choice(tips)