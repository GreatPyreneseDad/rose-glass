#!/usr/bin/env python3
"""
SoulMath Fear Elevation System - Command Line Interface
Interactive CLI for fear analysis and elevation journey
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Optional
import json
import time

from core.fear_engine import FearElevationEngine, FearInstance
from core.coherence_tracker import CoherenceTracker
from core.elevation_calculator import ElevationCalculator, DescentPoint
from agents.fear_analyzer import FearAnalyzer


class FearElevationCLI:
    """Interactive command-line interface for the Fear Elevation System."""
    
    def __init__(self):
        # Initialize core components
        self.engine = FearElevationEngine()
        self.coherence_tracker = CoherenceTracker()
        self.elevation_calc = ElevationCalculator()
        self.fear_analyzer = FearAnalyzer()
        
        # Session state
        self.current_descent = []
        self.session_start = datetime.now()
        self.in_descent = False
        
    def run(self):
        """Main CLI loop."""
        self.print_welcome()
        
        while True:
            try:
                self.show_status()
                command = input("\n→ ").strip().lower()
                
                if command in ['quit', 'exit', 'q']:
                    self.save_session()
                    print("\n🌟 Fear transforms. Elevation remains. Until next time.\n")
                    break
                    
                self.process_command(command)
                
            except KeyboardInterrupt:
                print("\n\n⚡ Sudden exit. Coherence maintained.\n")
                break
            except Exception as e:
                print(f"\n⚠️  Error: {e}\n")
                
    def print_welcome(self):
        """Print welcome message and introduction."""
        print("\n" + "="*60)
        print("🌊 SOULMATH FEAR ELEVATION SYSTEM 🗻")
        print("="*60)
        print("\nTheorem: 'Fear as the Architect of Elevation'")
        print("Only by embracing our deepest fears can we reach our greatest heights.")
        print("\nH ∝ ∫(descent→truth) F(x)dx")
        print("\nCommands:")
        print("  analyze   - Analyze your fears")
        print("  descend   - Begin fear descent")
        print("  embrace   - Embrace current fear")
        print("  status    - View coherence status")
        print("  breathe   - Coherence breathing")
        print("  insight   - Receive wisdom")
        print("  history   - View journey")
        print("  help      - Show all commands")
        print("  quit      - Exit system")
        print("="*60)
        
    def show_status(self):
        """Display current coherence and elevation status."""
        report = self.coherence_tracker.get_coherence_report()
        
        print(f"\n📊 Coherence: {report['current_psi']:.2f} ({report['current_state']})")
        print(f"📈 Elevation: {self.engine.elevation_height:.2f}m")
        print(f"🎯 Stability: {report['stability_score']:.1%}")
        
        if self.in_descent:
            print(f"🔽 In Descent - Depth: {self.current_descent[-1].depth:.2f}" if self.current_descent else "🔽 In Descent")
            
    def process_command(self, command: str):
        """Process user commands."""
        parts = command.split(maxsplit=1)
        cmd = parts[0] if parts else ""
        args = parts[1] if len(parts) > 1 else ""
        
        commands = {
            'analyze': self.cmd_analyze,
            'a': self.cmd_analyze,
            'descend': self.cmd_descend,
            'd': self.cmd_descend,
            'embrace': self.cmd_embrace,
            'e': self.cmd_embrace,
            'status': self.cmd_status,
            's': self.cmd_status,
            'breathe': self.cmd_breathe,
            'b': self.cmd_breathe,
            'insight': self.cmd_insight,
            'i': self.cmd_insight,
            'history': self.cmd_history,
            'h': self.cmd_history,
            'help': self.cmd_help,
            '?': self.cmd_help,
        }
        
        if cmd in commands:
            commands[cmd](args)
        else:
            print("Unknown command. Type 'help' for available commands.")
            
    def cmd_analyze(self, args: str):
        """Analyze fears command."""
        if not args:
            print("\n🔍 Describe your fear (or what troubles you):")
            args = input("   ")
            
        print("\n🌀 Analyzing fear patterns...")
        
        # Get current context
        context = {
            'coherence': self.coherence_tracker.current_psi,
            'in_descent': self.in_descent
        }
        
        # Analyze
        result = self.fear_analyzer.analyze_fear(args, context)
        
        # Display results
        print(f"\n📋 Analysis Complete")
        print(f"━━━━━━━━━━━━━━━━━━━━")
        
        if result.primary_fear:
            print(f"\n🎯 Primary Fear: {result.primary_fear.pattern_type.replace('_', ' ').title()}")
            print(f"   Depth: {result.primary_fear.depth:.1f} | Potential: {result.primary_fear.transformative_potential:.1f}")
            print(f"\n💭 Guidance: {result.primary_fear.guidance}")
            
        print(f"\n🗺️  Fear Landscape:")
        for fear_type, intensity in result.fear_landscape.items():
            bar = "█" * int(intensity * 10)
            print(f"   {fear_type.replace('_', ' ').title():<20} {bar} {intensity:.1f}")
            
        if result.warnings:
            print(f"\n⚠️  Warnings:")
            for warning in result.warnings:
                print(f"   • {warning}")
                
        print(f"\n🧭 Approach: {result.recommended_approach}")
        
    def cmd_descend(self, args: str):
        """Begin or continue descent."""
        if not self.in_descent:
            print("\n🌊 Beginning descent into fear...")
            print("   (Type 'embrace' when you touch truth)")
            self.in_descent = True
            self.current_descent = []
            
            # Initial coherence check
            self.coherence_tracker.update_coherence(
                -0.05, "descent_begin", "voluntary_descent"
            )
        else:
            # Deepen descent
            current_depth = self.current_descent[-1].depth if self.current_descent else 0.1
            new_depth = min(1.0, current_depth + 0.1)
            
            print(f"\n🔽 Descending deeper... (depth: {new_depth:.1f})")
            
            # Get fear intensity
            print("   How intense is the fear now? (1-10):")
            try:
                intensity = float(input("   ")) / 10
            except:
                intensity = 0.5
                
            # Create descent point
            point = DescentPoint(
                depth=new_depth,
                timestamp=datetime.now(),
                fear_intensity=intensity,
                coherence=self.coherence_tracker.current_psi
            )
            self.current_descent.append(point)
            
            # Update coherence
            self.coherence_tracker.update_coherence(
                -0.1 * new_depth, "descent_deeper", f"depth_{new_depth:.1f}"
            )
            
            # Check if approaching truth threshold
            if new_depth >= 0.85:
                print("\n✨ You approach the threshold of truth...")
                print("   The fear transforms. What do you see?")
                
    def cmd_embrace(self, args: str):
        """Embrace the current fear."""
        if not self.in_descent or not self.current_descent:
            print("\n❌ No active descent. Use 'descend' first.")
            return
            
        print("\n🫂 Embracing the fear...")
        time.sleep(1)
        
        # Get the deepest point
        deepest = max(self.current_descent, key=lambda p: p.depth)
        
        # Calculate elevation
        elevation_result = self.elevation_calc.calculate_elevation(
            self.current_descent,
            fear_type='default'
        )
        
        # Create fear instance
        fear = FearInstance(
            fear_type="descended_fear",
            depth=deepest.depth,
            description="Fear faced through descent",
            timestamp=datetime.now()
        )
        
        # Embrace through engine
        delta_psi, elevation = self.engine.embrace_fear(fear)
        
        # Update coherence
        self.coherence_tracker.update_coherence(
            delta_psi, "fear_embraced", f"elevation_{elevation:.1f}"
        )
        
        # Display results
        print(f"\n🌟 TRANSFORMATION COMPLETE")
        print(f"━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"📊 Coherence gained: +{delta_psi:.2f}")
        print(f"🏔️  Elevation achieved: {elevation:.1f}m")
        print(f"📈 Total elevation: {self.engine.elevation_height:.1f}m")
        print(f"✨ {self.engine.generate_insight()}")
        
        # End descent
        self.in_descent = False
        self.current_descent = []
        
    def cmd_status(self, args: str):
        """Show detailed status."""
        report = self.coherence_tracker.get_coherence_report()
        
        print(f"\n📊 SYSTEM STATUS")
        print(f"━━━━━━━━━━━━━━━")
        print(f"Coherence (Ψ): {report['current_psi']:.3f}")
        print(f"State: {report['current_state']}")
        print(f"Stability: {report['stability_score']:.1%}")
        print(f"Trend: {report['trend']}")
        print(f"Resonance: {report['resonance_frequency']:.2f}")
        print(f"\nElevation: {self.engine.elevation_height:.2f}m")
        print(f"Fears Faced: {len([f for f in self.engine.fear_field if f.embraced])}")
        print(f"Session Time: {(datetime.now() - self.session_start).seconds // 60} minutes")
        
        # Predictions
        predictions = self.coherence_tracker.predict_trajectory(3)
        print(f"\n🔮 Predicted Trajectory:")
        for i, pred in enumerate(predictions):
            print(f"   +{i+1}: {pred:.2f}")
            
    def cmd_breathe(self, args: str):
        """Coherence breathing exercise."""
        print("\n🌬️  Coherence Breathing")
        print("   Follow the rhythm...")
        print()
        
        for i in range(5):
            print("   Inhale... ", end='', flush=True)
            time.sleep(4)
            print("Hold... ", end='', flush=True)
            time.sleep(4)
            print("Exhale... ", end='', flush=True)
            time.sleep(4)
            print("Hold...")
            time.sleep(4)
            
        # Boost coherence slightly
        self.coherence_tracker.update_coherence(
            0.1, "breathing_practice", "coherence_breathing"
        )
        
        print("\n✨ Coherence increased. You are ready.")
        
    def cmd_insight(self, args: str):
        """Generate insight based on current state."""
        insight = self.engine.generate_insight()
        
        print(f"\n💭 INSIGHT")
        print(f"━━━━━━━━━")
        print(f"{insight}")
        
        # Additional wisdom based on state
        if self.coherence_tracker.current_psi < 0.5:
            print("\n🕯️  When coherence fragments, return to breath.")
        elif self.engine.elevation_height > 5:
            print("\n🏔️  You have climbed far. Rest in your achievement.")
        elif self.in_descent:
            print("\n🌊 The descent continues. Trust the process.")
            
    def cmd_history(self, args: str):
        """View journey history."""
        print(f"\n📜 JOURNEY HISTORY")
        print(f"━━━━━━━━━━━━━━━━")
        
        journey = self.engine.export_journey()
        
        print(f"Fears Faced: {journey['fears_faced']}")
        print(f"Fears Embraced: {journey['fears_embraced']}")
        print(f"Current Elevation: {journey['elevation_height']:.2f}m")
        print(f"Coherence: {journey['coherence_state']:.2f}")
        
        if journey['journey_history']:
            print(f"\n🎯 Recent Transformations:")
            for event in journey['journey_history'][-5:]:
                time_str = event['timestamp'].strftime('%H:%M')
                print(f"   {time_str} - {event['fear_type']} → +{event['elevation']:.1f}m")
                
    def cmd_help(self, args: str):
        """Show help."""
        print("\n📖 COMMANDS")
        print("━━━━━━━━━━")
        print("analyze [text] - Analyze your fears")
        print("descend       - Begin/continue fear descent")
        print("embrace       - Embrace current fear")
        print("status        - View detailed status")
        print("breathe       - Coherence breathing")
        print("insight       - Receive insight")
        print("history       - View your journey")
        print("help          - Show this help")
        print("quit          - Exit system")
        print("\n💡 Tip: Single letters work too (a, d, e, s, b, i, h)")
        
    def save_session(self):
        """Save session data."""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'duration': (datetime.now() - self.session_start).seconds,
            'engine_state': self.engine.export_journey(),
            'coherence_state': self.coherence_tracker.export_data(),
            'elevation_history': self.elevation_calc.export_calculations()
        }
        
        # Save to data directory
        os.makedirs('data', exist_ok=True)
        filename = f"data/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
            
        print(f"\n💾 Session saved: {filename}")


def main():
    """Main entry point."""
    cli = FearElevationCLI()
    cli.run()


if __name__ == '__main__':
    main()