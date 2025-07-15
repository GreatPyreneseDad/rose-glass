"""
Example usage of the GCT Learning Path Generator
Demonstrates creating modules and generating personalized paths
"""

import sys
sys.path.append('src')

from repository import CourseRepository, Module
from metadata_extractor import MetadataExtractor
from gct_engine import LearningGCTEngine
from path_optimizer import LearningPathOptimizer
import pandas as pd
import json


def create_sample_modules():
    """Create sample learning modules for demonstration"""
    
    modules = [
        # Python Basics Track
        Module(
            module_id="py_intro",
            title="Introduction to Python",
            description="Learn Python basics: variables, data types, and simple operations",
            topic_tags=["python", "programming", "basics"],
            difficulty=0.1,
            duration_minutes=30,
            prerequisites=[],
            learning_objectives=["understand variables", "use basic data types", "write simple programs"],
            content_type="video",
            cognitive_load=0.2
        ),
        Module(
            module_id="py_control_flow",
            title="Control Flow in Python",
            description="Master if statements, loops, and flow control",
            topic_tags=["python", "programming", "control_flow"],
            difficulty=0.25,
            duration_minutes=45,
            prerequisites=["py_intro"],
            learning_objectives=["implement conditional logic", "use for and while loops", "understand flow control"],
            content_type="interactive",
            cognitive_load=0.4
        ),
        Module(
            module_id="py_functions",
            title="Python Functions",
            description="Create reusable code with functions",
            topic_tags=["python", "programming", "functions"],
            difficulty=0.35,
            duration_minutes=60,
            prerequisites=["py_control_flow"],
            learning_objectives=["define functions", "use parameters and return values", "understand scope"],
            content_type="mixed",
            cognitive_load=0.5
        ),
        Module(
            module_id="py_data_structures",
            title="Data Structures",
            description="Work with lists, dictionaries, and sets",
            topic_tags=["python", "programming", "data_structures"],
            difficulty=0.4,
            duration_minutes=75,
            prerequisites=["py_functions"],
            learning_objectives=["use lists effectively", "implement dictionaries", "understand sets"],
            content_type="interactive",
            cognitive_load=0.6
        ),
        
        # Data Science Track
        Module(
            module_id="ds_numpy",
            title="NumPy Fundamentals",
            description="Numerical computing with NumPy arrays",
            topic_tags=["python", "data_science", "numpy", "mathematics"],
            difficulty=0.45,
            duration_minutes=60,
            prerequisites=["py_data_structures"],
            learning_objectives=["create numpy arrays", "perform array operations", "use numpy functions"],
            content_type="video",
            cognitive_load=0.6
        ),
        Module(
            module_id="ds_pandas",
            title="Data Analysis with Pandas",
            description="Manipulate and analyze data using Pandas",
            topic_tags=["python", "data_science", "pandas", "data_analysis"],
            difficulty=0.5,
            duration_minutes=90,
            prerequisites=["ds_numpy"],
            learning_objectives=["create DataFrames", "filter and transform data", "analyze datasets"],
            content_type="interactive",
            cognitive_load=0.65
        ),
        Module(
            module_id="ds_visualization",
            title="Data Visualization",
            description="Create compelling visualizations with Matplotlib and Seaborn",
            topic_tags=["python", "data_science", "visualization", "matplotlib"],
            difficulty=0.45,
            duration_minutes=60,
            prerequisites=["ds_pandas"],
            learning_objectives=["create basic plots", "customize visualizations", "tell stories with data"],
            content_type="mixed",
            cognitive_load=0.5
        ),
        
        # Machine Learning Track
        Module(
            module_id="ml_intro",
            title="Introduction to Machine Learning",
            description="Core concepts and principles of ML",
            topic_tags=["machine_learning", "data_science", "ai", "theory"],
            difficulty=0.55,
            duration_minutes=45,
            prerequisites=["ds_pandas"],
            learning_objectives=["understand ML concepts", "identify ML problems", "know ML workflow"],
            content_type="video",
            cognitive_load=0.7
        ),
        Module(
            module_id="ml_supervised",
            title="Supervised Learning",
            description="Classification and regression algorithms",
            topic_tags=["machine_learning", "supervised_learning", "algorithms"],
            difficulty=0.65,
            duration_minutes=120,
            prerequisites=["ml_intro", "ds_numpy"],
            learning_objectives=["implement classifiers", "build regression models", "evaluate performance"],
            content_type="interactive",
            cognitive_load=0.75
        ),
        Module(
            module_id="ml_neural_nets",
            title="Neural Networks Basics",
            description="Introduction to deep learning",
            topic_tags=["machine_learning", "deep_learning", "neural_networks"],
            difficulty=0.75,
            duration_minutes=150,
            prerequisites=["ml_supervised"],
            learning_objectives=["understand neural networks", "implement basic networks", "use backpropagation"],
            content_type="mixed",
            cognitive_load=0.85
        )
    ]
    
    return modules


def demonstrate_learning_paths():
    """Demonstrate the learning path generation system"""
    
    print("🎓 GCT Learning Path Generator Demo\n")
    
    # Initialize components
    print("📚 Setting up course repository...")
    repo_config = {'type': 'json', 'path': 'data/demo_modules.json'}
    repo = CourseRepository(repo_config)
    
    # Add sample modules
    modules = create_sample_modules()
    for module in modules:
        repo.add_module(module)
    print(f"✅ Added {len(modules)} learning modules\n")
    
    # Initialize metadata extractor
    print("🔍 Extracting module metadata...")
    extractor = MetadataExtractor()
    
    # Extract features
    module_list = repo.list_modules()
    topic_vectors_df = extractor.extract_topic_vectors(module_list)
    difficulty_df = extractor.assign_difficulty_scores(module_list)
    
    # Merge metadata
    metadata_df = pd.merge(topic_vectors_df, difficulty_df, on='module_id')
    print("✅ Metadata extraction complete\n")
    
    # Initialize GCT engine
    print("⚡ Initializing GCT coherence engine...")
    gct_engine = LearningGCTEngine()
    
    # Calculate coherence scores
    score_matrix = gct_engine.score_transitions(metadata_df)
    print("✅ Coherence matrix calculated\n")
    
    # Create learning paths for different learners
    demo_learners = [
        {
            'learner_id': 'beginner_ben',
            'skill_level': 0.2,
            'learning_style': 'visual',
            'goals': ['learn python programming'],
            'interests': ['web_development'],
            'constraints': {'max_daily_minutes': 60}
        },
        {
            'learner_id': 'data_diana',
            'skill_level': 0.5,
            'learning_style': 'interactive',
            'goals': ['become data scientist', 'analyze business data'],
            'interests': ['data_science', 'visualization'],
            'constraints': {'max_daily_minutes': 90}
        },
        {
            'learner_id': 'ml_mike',
            'skill_level': 0.7,
            'learning_style': 'reading',
            'goals': ['master machine learning', 'build AI applications'],
            'interests': ['machine_learning', 'deep_learning'],
            'constraints': {'max_daily_minutes': 120}
        }
    ]
    
    # Generate paths for each learner
    for learner in demo_learners:
        print(f"\n{'='*60}")
        print(f"👤 Learner: {learner['learner_id']}")
        print(f"   Skill Level: {learner['skill_level']}")
        print(f"   Goals: {', '.join(learner['goals'])}")
        print(f"   Daily Time: {learner['constraints']['max_daily_minutes']} minutes")
        
        # Create optimizer for this learner
        optimizer = LearningPathOptimizer(gct_engine, learner)
        
        # Create transition graph
        module_graph = gct_engine.create_transition_graph(score_matrix)
        
        # Generate learning path
        if learner['learner_id'] == 'beginner_ben':
            start, target = 'py_intro', 'py_data_structures'
        elif learner['learner_id'] == 'data_diana':
            start, target = 'py_functions', 'ds_visualization'
        else:
            start, target = 'ds_pandas', 'ml_neural_nets'
        
        print(f"\n🎯 Generating path from '{start}' to '{target}'...")
        
        path = optimizer.build_path(start, target, module_graph, metadata_df)
        
        print(f"\n✨ Optimal Learning Path (Coherence: {path.total_coherence:.2f}):")
        print(f"   Total Duration: {path.estimated_duration} minutes ({path.estimated_duration/60:.1f} hours)")
        
        # Display path details
        for i, module_id in enumerate(path.modules):
            module = next(m for m in module_list if m['module_id'] == module_id)
            print(f"\n   {i+1}. {module['title']}")
            print(f"      Difficulty: {'▪' * int(path.difficulty_curve[i] * 10)} ({path.difficulty_curve[i]:.2f})")
            print(f"      Duration: {module['duration_minutes']} min")
            print(f"      Topics: {', '.join(module['topic_tags'])}")
            
            # Show coherence for transitions
            if i < len(path.modules) - 1:
                next_module = path.modules[i + 1]
                coherence = score_matrix.loc[module_id, next_module]
                print(f"      → Transition coherence to next: {coherence:.2f}")
        
        # Generate alternative paths
        print(f"\n🔄 Alternative Paths:")
        alternatives = optimizer.generate_alternative_paths(
            start, target, module_graph, metadata_df, n_alternatives=2
        )
        
        for j, alt_path in enumerate(alternatives[1:], 1):  # Skip first (same as optimal)
            print(f"   Alternative {j}: {' → '.join(alt_path.modules)}")
            print(f"   Coherence: {alt_path.total_coherence:.2f}, Duration: {alt_path.estimated_duration} min")
    
    print(f"\n{'='*60}")
    print("\n✅ Demo complete! Learning paths generated successfully.")
    
    # Save results
    results = {
        'modules': [m.to_dict() for m in modules],
        'learners': demo_learners,
        'coherence_matrix': score_matrix.to_dict()
    }
    
    with open('data/demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n📁 Results saved to data/demo_results.json")


def demonstrate_feedback_adaptation():
    """Show how the system adapts to learner feedback"""
    
    print("\n\n📊 Demonstrating Feedback Adaptation\n")
    
    # Simulate learner completing modules and providing feedback
    learner = {
        'learner_id': 'adaptive_alice',
        'skill_level': 0.4,
        'learning_style': 'balanced',
        'goals': ['improve programming skills'],
        'constraints': {}
    }
    
    # Initialize optimizer
    gct_engine = LearningGCTEngine()
    optimizer = LearningPathOptimizer(gct_engine, learner)
    
    # Simulate feedback
    feedback_samples = [
        {
            'module_id': 'py_control_flow',
            'completion_time': 55,  # Took longer than expected
            'difficulty_rating': 4.5,  # Found it hard
            'engagement_rating': 3.0,  # Medium engagement
            'quiz_score': 0.65,  # Struggled a bit
            'would_recommend': True
        },
        {
            'module_id': 'py_functions',
            'completion_time': 45,  # Faster than expected
            'difficulty_rating': 2.5,  # Found it easier
            'engagement_rating': 5.0,  # Very engaged
            'quiz_score': 0.95,  # Did great
            'would_recommend': True
        }
    ]
    
    print("📝 Processing learner feedback...")
    for feedback in feedback_samples:
        print(f"\n   Module: {feedback['module_id']}")
        print(f"   Difficulty: {feedback['difficulty_rating']}/5")
        print(f"   Quiz Score: {feedback['quiz_score']*100:.0f}%")
        
        # Update optimizer
        optimizer.update_on_feedback(feedback['module_id'], feedback)
    
    print("\n✅ Optimizer adapted based on feedback")
    print(f"   New skill level: {optimizer.learner_profile['skill_level']:.2f}")
    print(f"   Preferred duration: {optimizer.learner_profile['constraints'].get('preferred_duration', 30):.0f} min")
    print(f"   Optimal difficulty step: {optimizer.gct_engine.params['optimal_difficulty_step']:.3f}")


if __name__ == "__main__":
    # Create data directory
    import os
    os.makedirs('data', exist_ok=True)
    
    # Run demonstrations
    demonstrate_learning_paths()
    demonstrate_feedback_adaptation()
    
    print("\n\n🎉 All demonstrations complete!")
    print("💡 Try running the API with: python src/api.py")
    print("🌐 Then visit http://localhost:8000/docs for interactive API docs")