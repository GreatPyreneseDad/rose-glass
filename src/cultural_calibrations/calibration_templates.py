"""
Cultural Calibration Templates for Community Development
=======================================================

These templates provide starting points for communities to develop
their own calibrations. They should be refined through deep collaboration
with cultural practitioners and knowledge holders.
"""

from src.core.rose_glass_v2 import CulturalCalibration


class CalibrationTemplates:
    """Templates for cultural calibrations awaiting community refinement"""
    
    @staticmethod
    def create_ubuntu_philosophy() -> CulturalCalibration:
        """
        Template for Ubuntu philosophy calibration.
        Needs validation from African philosophy practitioners.
        """
        return CulturalCalibration(
            name="Ubuntu Philosophy",
            description="For African communalist philosophy - 'I am because we are'",
            km=0.18,  # Lower threshold - emotion flows freely in community
            ki=0.75,  # Moderate inhibition - balance individual/collective
            coupling_strength=0.28,  # High coupling - wisdom and emotion intertwined
            expected_patterns={
                'reasoning': 'Relational, consensus-building, proverb-based',
                'moral_expression': 'Embedded in community harmony and restoration',
                'social_architecture': 'Deeply collective - individual exists through others',
                'wisdom_integration': 'Ancestral wisdom, oral tradition, lived experience'
            },
            breathing_pattern="Call and response, rhythmic affirmation",
            temporal_context="Both timeless and contemporary African thought",
            philosophical_tradition="Ubuntu/Hunhu communalist philosophy"
        )
    
    @staticmethod
    def create_daoist_contemplative() -> CulturalCalibration:
        """
        Template for Daoist contemplative texts.
        Needs validation from Daoist scholars and practitioners.
        """
        return CulturalCalibration(
            name="Daoist Contemplative",
            description="For texts emphasizing wu wei, flow, and natural harmony",
            km=0.35,  # Higher threshold - emotions flow like water, not forced
            ki=1.1,   # High inhibition - extremes naturally balance
            coupling_strength=0.06,  # Low coupling - each aspect flows independently
            expected_patterns={
                'reasoning': 'Paradoxical, flowing, nature-metaphor based',
                'moral_expression': 'Through non-action and natural harmony',
                'social_architecture': 'Individual as part of cosmic flow',
                'wisdom_integration': 'Experiential, ineffable, pointing beyond words'
            },
            breathing_pattern="Natural rhythm like flowing water",
            temporal_context="Classical and contemporary Daoist thought",
            philosophical_tradition="Daoist philosophy and practice"
        )
    
    @staticmethod
    def create_neurodivergent_direct() -> CulturalCalibration:
        """
        Template for direct neurodivergent communication styles.
        MUST be developed with neurodivergent communities.
        """
        return CulturalCalibration(
            name="Neurodivergent Direct Communication",
            description="For direct, literal communication styles (with consent)",
            km=0.20,  # Variable - emotions may be expressed differently
            ki=0.70,  # Less inhibition - direct expression valued
            coupling_strength=0.15,  # Moderate - connections may be explicit
            expected_patterns={
                'reasoning': 'Direct, literal, detailed, systematic',
                'moral_expression': 'Clear statements of values and boundaries',
                'social_architecture': 'Explicit rather than implied connections',
                'wisdom_integration': 'Through special interests and pattern recognition'
            },
            breathing_pattern="Variable - may include info-dumping or precise pauses",
            temporal_context="Contemporary neurodiversity movement",
            philosophical_tradition="Neurodiversity paradigm"
        )
    
    @staticmethod
    def create_quechua_andean() -> CulturalCalibration:
        """
        Template for Quechua/Andean indigenous thought.
        Needs development with Andean communities.
        """
        return CulturalCalibration(
            name="Quechua-Andean Cosmovision",
            description="For Andean indigenous philosophy of ayni and reciprocity",
            km=0.16,  # Lower - emotions integrated with cosmic balance
            ki=0.85,  # Moderate - extremes balanced through ayni
            coupling_strength=0.30,  # High - all elements interconnected
            expected_patterns={
                'reasoning': 'Complementary dualism, reciprocal, cyclical',
                'moral_expression': 'Through reciprocity (ayni) and balance',
                'social_architecture': 'Ayllu (community) centered, vertical ecology',
                'wisdom_integration': 'Living knowledge, ancestral-future time'
            },
            breathing_pattern="Cyclical like seasons, paired opposites",
            temporal_context="Pachakutik - cyclical time of transformation",
            philosophical_tradition="Andean cosmovision and philosophy"
        )
    
    @staticmethod
    def create_zen_koan() -> CulturalCalibration:
        """
        Template for Zen koan tradition.
        Needs validation from Zen teachers.
        """
        return CulturalCalibration(
            name="Zen Koan Tradition",
            description="For paradoxical teachings that shatter conceptual thinking",
            km=0.40,  # Higher - emotions arise but don't stick
            ki=0.95,  # High - preventing fixation
            coupling_strength=0.04,  # Very low - dimensions may contradict
            expected_patterns={
                'reasoning': 'Paradoxical, sudden, beyond logic',
                'moral_expression': 'Through direct pointing, not rules',
                'social_architecture': 'Teacher-student transmission',
                'wisdom_integration': 'Sudden illumination, not gradual accumulation'
            },
            breathing_pattern="Sudden shouts and long silences",
            temporal_context="Timeless present moment",
            philosophical_tradition="Zen Buddhism - koan tradition"
        )
    
    @classmethod
    def get_all_templates(cls) -> dict:
        """Return all available templates"""
        return {
            'ubuntu_philosophy': cls.create_ubuntu_philosophy(),
            'daoist_contemplative': cls.create_daoist_contemplative(),
            'neurodivergent_direct': cls.create_neurodivergent_direct(),
            'quechua_andean': cls.create_quechua_andean(),
            'zen_koan': cls.create_zen_koan()
        }
    
    @staticmethod
    def important_note() -> str:
        """Critical reminder about these templates"""
        return """
        IMPORTANT: These templates are starting points only!
        
        They MUST be refined through deep collaboration with communities.
        No calibration should be used without community validation.
        
        The parameters here are educated guesses that need correction
        from actual practitioners and knowledge holders.
        
        Using these templates without community partnership would be
        a form of digital colonialism - exactly what Rose Glass opposes.
        
        Proper process:
        1. Reach out to community organizations
        2. Establish partnership agreements
        3. Collaboratively refine parameters
        4. Test with community texts
        5. Validate interpretations
        6. Share benefits back to community
        """


# Example of how to properly develop a calibration
def develop_calibration_with_community():
    """
    Example process for ethical calibration development
    """
    # Step 1: Start with template
    template = CalibrationTemplates.create_ubuntu_philosophy()
    
    # Step 2: Community workshop (simulated)
    community_feedback = {
        'km': "Should be even lower - 0.12 - emotions flow very freely",
        'breathing': "Add call-response patterns, praise singing rhythms",
        'reasoning': "Include dreams and visions as valid reasoning",
        'warning': "Never use this to judge African expression by Western standards"
    }
    
    # Step 3: Revise based on feedback
    # (In reality, this would be iterative with many rounds)
    
    # Step 4: Test with community texts
    # Step 5: Validate interpretations
    # Step 6: Only deploy with explicit permission
    
    return "Calibration development is a journey, not a destination"


if __name__ == "__main__":
    print(CalibrationTemplates.important_note())
    print("\nAvailable templates (awaiting community development):")
    for name, calibration in CalibrationTemplates.get_all_templates().items():
        print(f"\n- {calibration.name}")
        print(f"  {calibration.description}")
        print("  STATUS: Needs community validation")