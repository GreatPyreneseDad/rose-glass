"""
Collaborative Creativity Enhancement using GCT
Team coherence synchronization and creative synergy optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import itertools

from .creative_flow_engine import CreativeFlowEngine, CreativeState, CreativeMetrics


@dataclass
class TeamMember:
    """Individual team member profile"""
    id: str
    name: str
    creative_profile: Dict[str, float]  # Preferred states and strengths
    coherence_history: List[float]
    state_history: List[CreativeState]
    skills: Set[str]
    collaboration_preferences: Dict[str, float]


@dataclass
class CollaborationMetrics:
    """Metrics for collaborative creativity"""
    team_coherence: float  # Overall team coherence
    synchronization_index: float  # How synchronized the team is
    diversity_score: float  # Beneficial diversity vs harmful fragmentation
    creative_friction: float  # Productive tension vs destructive conflict
    emergence_potential: float  # Likelihood of emergent innovation
    role_clarity: float  # How well-defined roles are
    communication_flow: float  # Quality of idea exchange


@dataclass
class TeamComposition:
    """Optimal team composition recommendation"""
    members: List[TeamMember]
    predicted_synergy: float
    role_assignments: Dict[str, str]
    interaction_patterns: Dict[Tuple[str, str], float]
    warnings: List[str]


class CollaborativeCreativityOptimizer:
    """Optimize creative teams using coherence synchronization"""
    
    def __init__(self):
        self.engine = CreativeFlowEngine()
        
        # Collaboration parameters
        self.collab_params = {
            'min_team_size': 2,
            'max_team_size': 8,
            'sync_threshold': 0.6,  # Minimum synchronization for effective collaboration
            'diversity_optimum': 0.4,  # Optimal diversity level
            'friction_threshold': 0.3,  # Maximum productive friction
        }
        
        # Creative roles based on coherence profiles
        self.creative_roles = {
            'explorer': {
                'primary_state': CreativeState.EXPLORATION,
                'coherence_profile': {'psi': 0.3, 'q': 0.7, 'rho': 0.4, 'f': 0.6}
            },
            'synthesizer': {
                'primary_state': CreativeState.INCUBATION,
                'coherence_profile': {'psi': 0.6, 'q': 0.4, 'rho': 0.8, 'f': 0.5}
            },
            'illuminator': {
                'primary_state': CreativeState.ILLUMINATION,
                'coherence_profile': {'psi': 0.8, 'q': 0.5, 'rho': 0.6, 'f': 0.4}
            },
            'refiner': {
                'primary_state': CreativeState.VERIFICATION,
                'coherence_profile': {'psi': 0.9, 'q': 0.3, 'rho': 0.7, 'f': 0.5}
            },
            'connector': {
                'primary_state': CreativeState.FLOW,
                'coherence_profile': {'psi': 0.7, 'q': 0.5, 'rho': 0.6, 'f': 0.8}
            }
        }
        
        # Interaction dynamics
        self.synergy_matrix = {
            ('explorer', 'synthesizer'): 0.8,
            ('explorer', 'illuminator'): 0.7,
            ('synthesizer', 'illuminator'): 0.9,
            ('illuminator', 'refiner'): 0.8,
            ('connector', 'any'): 0.7,  # Connectors work well with everyone
        }
        
    def analyze_team_coherence(self, 
                              team_members: List[TeamMember],
                              current_states: Dict[str, Dict]) -> CollaborationMetrics:
        """Analyze current team coherence and collaboration dynamics"""
        
        # Calculate individual coherences
        coherences = []
        states = []
        
        for member in team_members:
            if member.id in current_states:
                state_data = current_states[member.id]
                coherences.append(state_data['coherence'])
                states.append(state_data['creative_state'])
        
        # Team coherence (weighted average)
        team_coherence = np.mean(coherences) if coherences else 0
        
        # Synchronization index (inverse of variance)
        sync_index = 1 - np.std(coherences) if len(coherences) > 1 else 1
        
        # Diversity score
        diversity = self._calculate_diversity(team_members, states)
        
        # Creative friction
        friction = self._calculate_creative_friction(team_members, current_states)
        
        # Emergence potential
        emergence = self._calculate_emergence_potential(
            team_coherence, sync_index, diversity, friction
        )
        
        # Role clarity
        role_clarity = self._calculate_role_clarity(team_members, states)
        
        # Communication flow
        comm_flow = self._calculate_communication_flow(team_members, current_states)
        
        return CollaborationMetrics(
            team_coherence=team_coherence,
            synchronization_index=sync_index,
            diversity_score=diversity,
            creative_friction=friction,
            emergence_potential=emergence,
            role_clarity=role_clarity,
            communication_flow=comm_flow
        )
    
    def optimize_team_composition(self,
                                available_members: List[TeamMember],
                                project_requirements: Dict,
                                constraints: Optional[Dict] = None) -> TeamComposition:
        """Find optimal team composition for a creative project"""
        
        # Generate candidate teams
        min_size = constraints.get('min_size', self.collab_params['min_team_size'])
        max_size = constraints.get('max_size', self.collab_params['max_team_size'])
        
        best_team = None
        best_synergy = 0
        
        for size in range(min_size, max_size + 1):
            for team_combo in itertools.combinations(available_members, size):
                # Calculate predicted synergy
                synergy = self._predict_team_synergy(
                    list(team_combo), project_requirements
                )
                
                if synergy > best_synergy:
                    best_synergy = synergy
                    best_team = list(team_combo)
        
        if not best_team:
            return TeamComposition(
                members=[],
                predicted_synergy=0,
                role_assignments={},
                interaction_patterns={},
                warnings=["No suitable team composition found"]
            )
        
        # Assign roles
        role_assignments = self._assign_optimal_roles(best_team, project_requirements)
        
        # Predict interaction patterns
        interaction_patterns = self._predict_interactions(best_team)
        
        # Generate warnings
        warnings = self._generate_team_warnings(best_team, role_assignments)
        
        return TeamComposition(
            members=best_team,
            predicted_synergy=best_synergy,
            role_assignments=role_assignments,
            interaction_patterns=interaction_patterns,
            warnings=warnings
        )
    
    def real_time_collaboration_guide(self,
                                    team_members: List[TeamMember],
                                    current_states: Dict[str, Dict],
                                    project_phase: str) -> Dict:
        """Guide real-time creative collaboration"""
        
        # Analyze current collaboration metrics
        metrics = self.analyze_team_coherence(team_members, current_states)
        
        recommendations = []
        
        # Check synchronization
        if metrics.synchronization_index < self.collab_params['sync_threshold']:
            recommendations.append({
                'type': 'synchronization',
                'urgency': 'high',
                'action': 'Team synchronization needed',
                'suggestions': self._get_sync_suggestions(team_members, current_states)
            })
        
        # Check for role switches
        role_switches = self._suggest_role_switches(team_members, current_states, project_phase)
        if role_switches:
            recommendations.append({
                'type': 'role_optimization',
                'urgency': 'medium',
                'action': 'Consider role adjustments',
                'suggestions': role_switches
            })
        
        # Individual vs group work
        work_mode = self._determine_work_mode(metrics, project_phase)
        recommendations.append({
            'type': 'work_mode',
            'urgency': 'low',
            'action': f'Optimal work mode: {work_mode["mode"]}',
            'suggestions': work_mode['details']
        })
        
        # Breakthrough detection
        if metrics.emergence_potential > 0.7:
            recommendations.append({
                'type': 'breakthrough_alert',
                'urgency': 'high',
                'action': 'High emergence potential detected',
                'suggestions': [
                    'Maintain current team dynamics',
                    'Have capture tools ready',
                    'Minimize external interruptions'
                ]
            })
        
        return {
            'metrics': metrics,
            'recommendations': recommendations,
            'optimal_next_phase': self._predict_next_phase(metrics, project_phase)
        }
    
    def _calculate_diversity(self, members: List[TeamMember], states: List[CreativeState]) -> float:
        """Calculate beneficial diversity"""
        
        # Skill diversity
        all_skills = set()
        skill_overlap = 0
        
        for member in members:
            all_skills.update(member.skills)
        
        for m1, m2 in itertools.combinations(members, 2):
            skill_overlap += len(m1.skills.intersection(m2.skills))
        
        skill_diversity = 1 - (skill_overlap / (len(members) * len(all_skills) + 1))
        
        # State diversity
        unique_states = len(set(states)) if states else 1
        state_diversity = unique_states / len(CreativeState)
        
        # Profile diversity
        profile_diversity = 0
        for m1, m2 in itertools.combinations(members, 2):
            profile_distance = self._profile_distance(
                m1.creative_profile, m2.creative_profile
            )
            profile_diversity += profile_distance
        
        profile_diversity /= (len(members) * (len(members) - 1) / 2 + 1)
        
        # Weighted combination
        diversity = (
            skill_diversity * 0.3 +
            state_diversity * 0.3 +
            profile_diversity * 0.4
        )
        
        return min(1.0, diversity)
    
    def _calculate_creative_friction(self, 
                                   members: List[TeamMember],
                                   current_states: Dict[str, Dict]) -> float:
        """Calculate productive creative friction"""
        
        friction_components = []
        
        # Coherence differences
        coherences = [current_states.get(m.id, {}).get('coherence', 0.5) 
                     for m in members]
        coherence_variance = np.var(coherences) if len(coherences) > 1 else 0
        friction_components.append(coherence_variance * 2)
        
        # State conflicts
        states = [current_states.get(m.id, {}).get('creative_state', CreativeState.EXPLORATION) 
                 for m in members]
        
        conflicting_states = [
            (CreativeState.EXPLORATION, CreativeState.VERIFICATION),
            (CreativeState.FLOW, CreativeState.BLOCKED),
        ]
        
        conflict_count = 0
        for s1, s2 in itertools.combinations(states, 2):
            if (s1, s2) in conflicting_states or (s2, s1) in conflicting_states:
                conflict_count += 1
        
        state_friction = conflict_count / (len(members) * (len(members) - 1) / 2 + 1)
        friction_components.append(state_friction)
        
        # Preference conflicts
        pref_conflicts = 0
        for m1, m2 in itertools.combinations(members, 2):
            for pref_type in m1.collaboration_preferences:
                if pref_type in m2.collaboration_preferences:
                    diff = abs(m1.collaboration_preferences[pref_type] - 
                             m2.collaboration_preferences[pref_type])
                    pref_conflicts += diff
        
        pref_friction = pref_conflicts / (len(members) * 5)  # Assuming 5 preference types
        friction_components.append(pref_friction)
        
        return min(1.0, np.mean(friction_components))
    
    def _calculate_emergence_potential(self,
                                     team_coherence: float,
                                     sync_index: float,
                                     diversity: float,
                                     friction: float) -> float:
        """Calculate potential for emergent creative breakthroughs"""
        
        # Optimal conditions for emergence
        coherence_factor = team_coherence * (1 - abs(team_coherence - 0.7))
        sync_factor = sync_index * (1 - abs(sync_index - 0.6))
        diversity_factor = diversity * (1 - abs(diversity - self.collab_params['diversity_optimum']))
        friction_factor = friction * (1 - friction / self.collab_params['friction_threshold'])
        
        # Emergence requires all factors in balance
        emergence = (
            coherence_factor * 0.3 +
            sync_factor * 0.2 +
            diversity_factor * 0.3 +
            friction_factor * 0.2
        )
        
        # Boost if all factors are in optimal range
        if all([
            0.6 <= team_coherence <= 0.8,
            0.5 <= sync_index <= 0.7,
            0.3 <= diversity <= 0.5,
            0.1 <= friction <= 0.3
        ]):
            emergence *= 1.2
        
        return min(1.0, emergence)
    
    def _calculate_role_clarity(self, members: List[TeamMember], states: List[CreativeState]) -> float:
        """Calculate how well-defined team roles are"""
        
        # Match members to ideal roles
        role_matches = []
        
        for member, state in zip(members, states):
            best_role_match = 0
            
            for role_name, role_profile in self.creative_roles.items():
                if state == role_profile['primary_state']:
                    # Calculate profile match
                    profile_match = 1 - self._profile_distance(
                        member.creative_profile,
                        role_profile['coherence_profile']
                    )
                    best_role_match = max(best_role_match, profile_match)
            
            role_matches.append(best_role_match)
        
        return np.mean(role_matches) if role_matches else 0
    
    def _calculate_communication_flow(self,
                                    members: List[TeamMember],
                                    current_states: Dict[str, Dict]) -> float:
        """Calculate quality of idea exchange"""
        
        # High social coherence (f) indicates good communication
        f_values = []
        for member in members:
            if member.id in current_states:
                f_value = current_states[member.id].get('components', {}).get('f', 0.5)
                f_values.append(f_value)
        
        avg_f = np.mean(f_values) if f_values else 0.5
        
        # State compatibility for communication
        states = [current_states.get(m.id, {}).get('creative_state', CreativeState.EXPLORATION) 
                 for m in members]
        
        compatible_count = 0
        total_pairs = 0
        
        for s1, s2 in itertools.combinations(states, 2):
            total_pairs += 1
            # Some states communicate better
            if {s1, s2} in [
                {CreativeState.EXPLORATION, CreativeState.EXPLORATION},
                {CreativeState.FLOW, CreativeState.FLOW},
                {CreativeState.ILLUMINATION, CreativeState.VERIFICATION}
            ]:
                compatible_count += 1
        
        state_compatibility = compatible_count / (total_pairs + 1)
        
        return (avg_f * 0.6 + state_compatibility * 0.4)
    
    def _predict_team_synergy(self,
                            team: List[TeamMember],
                            requirements: Dict) -> float:
        """Predict synergy for a potential team"""
        
        synergy_factors = []
        
        # Skill coverage
        required_skills = set(requirements.get('required_skills', []))
        team_skills = set()
        for member in team:
            team_skills.update(member.skills)
        
        skill_coverage = len(required_skills.intersection(team_skills)) / (len(required_skills) + 1)
        synergy_factors.append(skill_coverage)
        
        # Role diversity
        member_profiles = [m.creative_profile for m in team]
        role_assignments = self._assign_optimal_roles(team, requirements)
        role_diversity = len(set(role_assignments.values())) / len(team)
        synergy_factors.append(role_diversity)
        
        # Pairwise synergies
        pair_synergies = []
        for m1, m2 in itertools.combinations(team, 2):
            role1 = role_assignments.get(m1.id, 'explorer')
            role2 = role_assignments.get(m2.id, 'explorer')
            
            # Check synergy matrix
            synergy = self.synergy_matrix.get((role1, role2), 0.5)
            if synergy == 0:
                synergy = self.synergy_matrix.get((role2, role1), 0.5)
            
            pair_synergies.append(synergy)
        
        avg_pair_synergy = np.mean(pair_synergies) if pair_synergies else 0.5
        synergy_factors.append(avg_pair_synergy)
        
        # Historical performance
        historical_synergy = self._calculate_historical_synergy(team)
        synergy_factors.append(historical_synergy)
        
        return np.mean(synergy_factors)
    
    def _assign_optimal_roles(self,
                            team: List[TeamMember],
                            requirements: Dict) -> Dict[str, str]:
        """Assign optimal roles to team members"""
        
        assignments = {}
        
        # Priority roles from requirements
        priority_roles = requirements.get('priority_roles', list(self.creative_roles.keys()))
        
        # Calculate fit scores
        fit_scores = {}
        for member in team:
            member_scores = {}
            
            for role_name, role_profile in self.creative_roles.items():
                if role_name in priority_roles:
                    # Profile similarity
                    profile_fit = 1 - self._profile_distance(
                        member.creative_profile,
                        role_profile['coherence_profile']
                    )
                    
                    # Historical performance in similar states
                    state_history_fit = member.state_history.count(
                        role_profile['primary_state']
                    ) / (len(member.state_history) + 1)
                    
                    member_scores[role_name] = profile_fit * 0.7 + state_history_fit * 0.3
            
            fit_scores[member.id] = member_scores
        
        # Assign roles optimally (simplified Hungarian algorithm)
        assigned_roles = set()
        
        for _ in range(len(team)):
            best_fit = 0
            best_assignment = None
            
            for member_id, scores in fit_scores.items():
                if member_id not in assignments:
                    for role, score in scores.items():
                        if role not in assigned_roles and score > best_fit:
                            best_fit = score
                            best_assignment = (member_id, role)
            
            if best_assignment:
                assignments[best_assignment[0]] = best_assignment[1]
                assigned_roles.add(best_assignment[1])
        
        # Assign default roles to remaining members
        for member in team:
            if member.id not in assignments:
                assignments[member.id] = 'connector'  # Default versatile role
        
        return assignments
    
    def _predict_interactions(self, team: List[TeamMember]) -> Dict[Tuple[str, str], float]:
        """Predict interaction patterns between team members"""
        
        interactions = {}
        
        for m1, m2 in itertools.combinations(team, 2):
            # Base interaction probability on profile similarity
            profile_similarity = 1 - self._profile_distance(
                m1.creative_profile, m2.creative_profile
            )
            
            # Adjust based on collaboration preferences
            pref_compatibility = 0
            for pref in m1.collaboration_preferences:
                if pref in m2.collaboration_preferences:
                    pref_compatibility += 1 - abs(
                        m1.collaboration_preferences[pref] - 
                        m2.collaboration_preferences[pref]
                    )
            
            pref_compatibility /= (len(m1.collaboration_preferences) + 1)
            
            # Combine factors
            interaction_strength = profile_similarity * 0.6 + pref_compatibility * 0.4
            
            interactions[(m1.id, m2.id)] = interaction_strength
        
        return interactions
    
    def _generate_team_warnings(self,
                              team: List[TeamMember],
                              role_assignments: Dict[str, str]) -> List[str]:
        """Generate warnings about potential team issues"""
        
        warnings = []
        
        # Check for missing critical roles
        assigned_roles = set(role_assignments.values())
        critical_roles = {'explorer', 'synthesizer', 'refiner'}
        missing_roles = critical_roles - assigned_roles
        
        if missing_roles:
            warnings.append(f"Missing critical roles: {', '.join(missing_roles)}")
        
        # Check for too many of same role
        role_counts = defaultdict(int)
        for role in role_assignments.values():
            role_counts[role] += 1
        
        for role, count in role_counts.items():
            if count > len(team) / 2:
                warnings.append(f"Too many members in {role} role ({count}/{len(team)})")
        
        # Check for isolated members
        avg_interactions = defaultdict(float)
        interaction_patterns = self._predict_interactions(team)
        
        for (m1, m2), strength in interaction_patterns.items():
            avg_interactions[m1] += strength
            avg_interactions[m2] += strength
        
        for member_id, total_interaction in avg_interactions.items():
            avg = total_interaction / (len(team) - 1)
            if avg < 0.3:
                member_name = next(m.name for m in team if m.id == member_id)
                warnings.append(f"{member_name} may become isolated (low interaction probability)")
        
        # Check for potential conflicts
        high_friction_pairs = []
        for (m1_id, m2_id), strength in interaction_patterns.items():
            if strength < 0.2:
                m1_name = next(m.name for m in team if m.id == m1_id)
                m2_name = next(m.name for m in team if m.id == m2_id)
                high_friction_pairs.append(f"{m1_name}-{m2_name}")
        
        if high_friction_pairs:
            warnings.append(f"Potential friction between: {', '.join(high_friction_pairs)}")
        
        return warnings
    
    def _profile_distance(self, profile1: Dict[str, float], profile2: Dict[str, float]) -> float:
        """Calculate distance between two creative profiles"""
        
        keys = set(profile1.keys()).union(set(profile2.keys()))
        
        vec1 = [profile1.get(k, 0.5) for k in keys]
        vec2 = [profile2.get(k, 0.5) for k in keys]
        
        return cosine(vec1, vec2)
    
    def _get_sync_suggestions(self,
                            members: List[TeamMember],
                            current_states: Dict[str, Dict]) -> List[str]:
        """Get suggestions for team synchronization"""
        
        suggestions = []
        
        # Identify outliers
        coherences = [current_states.get(m.id, {}).get('coherence', 0.5) for m in members]
        mean_coherence = np.mean(coherences)
        
        for member in members:
            member_coherence = current_states.get(member.id, {}).get('coherence', 0.5)
            if abs(member_coherence - mean_coherence) > 0.3:
                if member_coherence < mean_coherence:
                    suggestions.append(f"Help {member.name} increase coherence through clarity exercises")
                else:
                    suggestions.append(f"{member.name} can mentor others in achieving flow")
        
        # General sync activities
        suggestions.extend([
            "5-minute team meditation or breathing exercise",
            "Share current challenges and blockers openly",
            "Quick round of appreciation for team contributions"
        ])
        
        return suggestions
    
    def _suggest_role_switches(self,
                             members: List[TeamMember],
                             current_states: Dict[str, Dict],
                             project_phase: str) -> List[Dict]:
        """Suggest role switches based on current states and project phase"""
        
        suggestions = []
        
        # Phase-appropriate roles
        phase_roles = {
            'ideation': ['explorer', 'connector'],
            'development': ['synthesizer', 'illuminator'],
            'refinement': ['refiner', 'illuminator'],
            'finalization': ['refiner', 'connector']
        }
        
        ideal_roles = phase_roles.get(project_phase, ['explorer'])
        
        for member in members:
            current_state = current_states.get(member.id, {}).get('creative_state')
            if current_state:
                # Find matching role for current state
                for role_name, role_profile in self.creative_roles.items():
                    if current_state == role_profile['primary_state']:
                        if role_name not in ideal_roles:
                            suggestions.append({
                                'member': member.name,
                                'current_role': role_name,
                                'suggested_role': ideal_roles[0],
                                'reason': f"{role_name} less effective in {project_phase} phase"
                            })
                        break
        
        return suggestions
    
    def _determine_work_mode(self, metrics: CollaborationMetrics, project_phase: str) -> Dict:
        """Determine optimal work mode (individual vs collaborative)"""
        
        # Factors favoring collaboration
        collab_score = (
            metrics.synchronization_index * 0.3 +
            metrics.communication_flow * 0.3 +
            (1 - metrics.creative_friction) * 0.2 +
            metrics.emergence_potential * 0.2
        )
        
        # Phase considerations
        collaborative_phases = ['ideation', 'review', 'integration']
        individual_phases = ['deep_work', 'refinement', 'exploration']
        
        if project_phase in collaborative_phases:
            collab_score *= 1.2
        elif project_phase in individual_phases:
            collab_score *= 0.8
        
        if collab_score > 0.6:
            return {
                'mode': 'collaborative',
                'details': [
                    'Team is well-synchronized for group work',
                    'Consider brainstorming or co-creation sessions',
                    'Use shared workspace or collaborative tools'
                ]
            }
        else:
            return {
                'mode': 'individual',
                'details': [
                    'Team members benefit from focused individual work',
                    'Schedule check-ins every 30-45 minutes',
                    'Prepare integration session after individual work'
                ]
            }
    
    def _predict_next_phase(self, metrics: CollaborationMetrics, current_phase: str) -> str:
        """Predict optimal next project phase based on team metrics"""
        
        phase_transitions = {
            'ideation': {
                'next': 'development',
                'conditions': {'team_coherence': 0.6, 'diversity_score': 0.3}
            },
            'development': {
                'next': 'refinement',
                'conditions': {'team_coherence': 0.7, 'synchronization_index': 0.6}
            },
            'refinement': {
                'next': 'finalization',
                'conditions': {'role_clarity': 0.7, 'creative_friction': 0.2}
            }
        }
        
        if current_phase in phase_transitions:
            transition = phase_transitions[current_phase]
            conditions_met = True
            
            for metric, threshold in transition['conditions'].items():
                if hasattr(metrics, metric):
                    if getattr(metrics, metric) < threshold:
                        conditions_met = False
                        break
            
            if conditions_met:
                return transition['next']
        
        return current_phase  # Stay in current phase
    
    def _calculate_historical_synergy(self, team: List[TeamMember]) -> float:
        """Calculate synergy based on historical performance"""
        
        # Simplified - would need actual historical collaboration data
        synergy_scores = []
        
        for m1, m2 in itertools.combinations(team, 2):
            # Check if coherence histories are correlated (indicating good sync)
            if len(m1.coherence_history) > 10 and len(m2.coherence_history) > 10:
                correlation, _ = pearsonr(
                    m1.coherence_history[-10:],
                    m2.coherence_history[-10:]
                )
                synergy_scores.append(abs(correlation))
        
        return np.mean(synergy_scores) if synergy_scores else 0.5


class CreativeProjectOrchestrator:
    """Orchestrate entire creative projects using GCT"""
    
    def __init__(self):
        self.collab_optimizer = CollaborativeCreativityOptimizer()
        self.project_phases = [
            'ideation', 'exploration', 'development', 
            'refinement', 'integration', 'finalization'
        ]
        
    def plan_creative_project(self,
                            project_goals: Dict,
                            available_members: List[TeamMember],
                            timeline: Dict) -> Dict:
        """Create comprehensive project plan based on GCT analysis"""
        
        project_plan = {
            'phases': [],
            'team_compositions': {},
            'milestones': [],
            'risk_factors': []
        }
        
        # Plan each phase
        for phase in self.project_phases:
            phase_requirements = self._get_phase_requirements(phase, project_goals)
            
            # Optimize team for this phase
            team_comp = self.collab_optimizer.optimize_team_composition(
                available_members,
                phase_requirements
            )
            
            project_plan['team_compositions'][phase] = team_comp
            
            # Estimate phase duration
            phase_duration = self._estimate_phase_duration(
                phase, team_comp, timeline
            )
            
            project_plan['phases'].append({
                'name': phase,
                'duration': phase_duration,
                'team': [m.name for m in team_comp.members],
                'key_roles': team_comp.role_assignments,
                'success_metrics': self._define_phase_metrics(phase)
            })
        
        # Identify risks
        project_plan['risk_factors'] = self._identify_project_risks(
            project_plan['team_compositions']
        )
        
        return project_plan
    
    def _get_phase_requirements(self, phase: str, goals: Dict) -> Dict:
        """Get requirements for each project phase"""
        
        base_requirements = {
            'ideation': {
                'required_skills': ['brainstorming', 'conceptualization'],
                'priority_roles': ['explorer', 'connector'],
                'team_size_range': (3, 6)
            },
            'development': {
                'required_skills': ['implementation', 'problem_solving'],
                'priority_roles': ['synthesizer', 'illuminator'],
                'team_size_range': (2, 8)
            },
            'refinement': {
                'required_skills': ['critical_analysis', 'optimization'],
                'priority_roles': ['refiner', 'illuminator'],
                'team_size_range': (2, 4)
            }
        }
        
        phase_req = base_requirements.get(phase, {})
        
        # Merge with project-specific requirements
        if 'required_skills' in goals:
            phase_req['required_skills'] = list(set(
                phase_req.get('required_skills', []) + goals['required_skills']
            ))
        
        return phase_req
    
    def _estimate_phase_duration(self,
                               phase: str,
                               team_comp: TeamComposition,
                               timeline: Dict) -> timedelta:
        """Estimate duration for each phase based on team composition"""
        
        base_durations = {
            'ideation': timedelta(days=3),
            'exploration': timedelta(days=5),
            'development': timedelta(days=14),
            'refinement': timedelta(days=7),
            'integration': timedelta(days=3),
            'finalization': timedelta(days=2)
        }
        
        base = base_durations.get(phase, timedelta(days=5))
        
        # Adjust based on team synergy
        synergy_factor = team_comp.predicted_synergy
        adjusted = base * (2 - synergy_factor)  # High synergy reduces time
        
        # Apply timeline constraints
        max_duration = timeline.get(f'{phase}_max', adjusted)
        
        return min(adjusted, max_duration)
    
    def _define_phase_metrics(self, phase: str) -> Dict[str, float]:
        """Define success metrics for each phase"""
        
        metrics = {
            'ideation': {
                'idea_quantity': 50,  # Target number of ideas
                'diversity_score': 0.7,  # Idea diversity target
                'team_engagement': 0.8
            },
            'development': {
                'progress_rate': 0.1,  # 10% daily progress
                'quality_score': 0.8,
                'team_coherence': 0.7
            },
            'refinement': {
                'improvement_rate': 0.15,
                'consistency_score': 0.9,
                'stakeholder_satisfaction': 0.8
            }
        }
        
        return metrics.get(phase, {})
    
    def _identify_project_risks(self, team_compositions: Dict[str, TeamComposition]) -> List[Dict]:
        """Identify potential risks across project phases"""
        
        risks = []
        
        # Check for team consistency
        all_members = set()
        for phase, comp in team_compositions.items():
            phase_members = set(m.id for m in comp.members)
            all_members.update(phase_members)
        
        # High turnover risk
        avg_participation = sum(
            len([p for p, c in team_compositions.items() 
                if any(m.id == member_id for m in c.members)])
            for member_id in all_members
        ) / len(all_members)
        
        if avg_participation < 0.5:
            risks.append({
                'type': 'high_turnover',
                'severity': 'medium',
                'mitigation': 'Ensure better team continuity across phases'
            })
        
        # Low synergy phases
        for phase, comp in team_compositions.items():
            if comp.predicted_synergy < 0.5:
                risks.append({
                    'type': 'low_synergy',
                    'phase': phase,
                    'severity': 'high',
                    'mitigation': f'Reconsider team composition for {phase} phase'
                })
        
        return risks