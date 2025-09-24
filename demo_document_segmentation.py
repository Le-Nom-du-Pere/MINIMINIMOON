"""
Document Segmentation Demo
=========================
Demonstrates the dual-criteria document segmentation system with various document types.
"""

from document_segmenter import DocumentSegmenter
import json


def main():
    """Run document segmentation demo with different text types."""
    
    # Initialize segmenter
    segmenter = DocumentSegmenter()
    
    print("=== Document Segmentation Demo ===\n")
    print(f"Configuration: {segmenter.target_char_min}-{segmenter.target_char_max} chars, {segmenter.target_sentences} sentences\n")
    
    # Demo texts of different types and lengths
    demo_texts = {
        "Technical Document": """
        Machine learning algorithms require careful preprocessing of input data. Data cleaning, normalization, and feature selection are essential steps in the pipeline. 
        The quality of preprocessing directly impacts model performance. Poor data quality leads to unreliable predictions and reduced accuracy.
        
        Feature engineering involves creating meaningful representations from raw data. Domain expertise guides the selection of relevant features. 
        Automated feature selection methods can help identify important variables. Cross-validation ensures robust feature selection across different data splits.
        
        Model selection depends on the specific problem requirements. Classification tasks use different algorithms than regression problems. 
        Performance metrics should align with business objectives. Regular evaluation prevents overfitting and ensures generalization.
        """,
        
        "News Article": """
        The latest breakthrough in renewable energy technology promises to revolutionize solar power generation. Scientists at the National Institute have developed new photovoltaic cells with unprecedented efficiency rates.
        The research team achieved 47% energy conversion efficiency in laboratory tests. This represents a significant improvement over current commercial solar panels.
        
        Industry experts believe this technology could reduce solar energy costs by 30%. Manufacturing challenges remain before commercial deployment becomes viable.
        The research received funding from multiple government agencies. International collaboration accelerated the development timeline significantly.
        
        Environmental impact assessments show positive results for large-scale deployment. Policy makers are reviewing regulatory frameworks for new technology adoption.
        """,
        
        "Academic Paper Abstract": """
        This study investigates the impact of social media usage on academic performance among university students. A comprehensive analysis was conducted using survey data from 2,500 students across multiple institutions.
        The research employed mixed-methods approaches combining quantitative surveys with qualitative interviews. Statistical analysis revealed significant correlations between social media usage patterns and GPA variations.
        
        Students spending more than three hours daily on social media platforms showed decreased academic performance. However, educational use of social media demonstrated positive learning outcomes.
        The findings suggest that context and purpose of social media use are critical factors. Recommendations include developing digital literacy programs for students.
        """,
        
        "Legal Document": """
        The plaintiff hereby alleges that the defendant violated the terms and conditions of the aforementioned contract dated January 15, 2024. Specific breaches include failure to deliver goods within the stipulated timeframe and substandard quality of delivered products.
        
        Evidence supporting these claims includes documented communications, delivery receipts, and quality inspection reports. The plaintiff seeks monetary damages totaling $150,000 plus legal fees and court costs.
        
        Furthermore, the plaintiff requests injunctive relief to prevent continued contract violations. The defendant's actions have caused substantial harm to the plaintiff's business operations and reputation.
        Immediate resolution through legal proceedings is necessary to prevent further damages. The court's intervention is respectfully requested to ensure justice and fair compensation.
        """,
        
        "Short Text": "This is a brief document. It has very few sentences. Segmentation should handle it gracefully.",
        
        "No Punctuation": "This text has no proper sentence boundaries it just keeps going and going without any clear breaks which makes segmentation challenging but the algorithm should still handle it reasonably well using fallback methods"
    }
    
    for doc_type, text in demo_texts.items():
        print(f"\n{'='*60}")
        print(f"Document Type: {doc_type}")
        print(f"Original length: {len(text)} characters")
        print('='*60)
        
        # Segment the document
        segments = segmenter.segment_document(text)
        
        print(f"Generated {len(segments)} segments:\n")
        
        for i, segment in enumerate(segments, 1):
            metrics = segment['metrics']
            print(f"Segment {i}:")
            print(f"  Text: {segment['text'][:100]}{'...' if len(segment['text']) > 100 else ''}")
            print(f"  Characters: {metrics.char_count}")
            print(f"  Sentences: {metrics.sentence_count}")  
            print(f"  Words: {metrics.word_count}")
            print(f"  Type: {metrics.segment_type}")
            print(f"  Meets char criteria: {segment['meets_char_criteria']}")
            print(f"  Meets sentence criteria: {segment['meets_sentence_criteria']}")
            print(f"  Coherence score: {metrics.semantic_coherence_score:.2f}")
            print()
        
        # Generate and display report
        report = segmenter.get_segmentation_report()
        
        print("Segmentation Quality Report:")
        print(f"  Overall quality score: {report['quality_indicators']['overall_quality_score']:.3f}")
        print(f"  Character range success: {report['summary']['char_range_success_rate']:.1f}%")
        print(f"  Sentence target success: {report['summary']['sentence_target_success_rate']:.1f}%")
        print(f"  Consistency score: {report['quality_indicators']['consistency_score']:.3f}")
        
        print("\nCharacter length distribution:")
        for bucket, count in report['character_analysis']['distribution'].items():
            print(f"  {bucket}: {count}")
            
        print("\nSentence count distribution:")
        for count, segments_num in report['sentence_analysis']['distribution'].items():
            print(f"  {count} sentences: {segments_num}")
    
    # Demonstrate different configurations
    print(f"\n{'='*80}")
    print("Configuration Comparison Demo")
    print('='*80)
    
    configurations = [
        {'name': 'Compact', 'target_char_min': 400, 'target_char_max': 600, 'target_sentences': 2},
        {'name': 'Standard', 'target_char_min': 700, 'target_char_max': 900, 'target_sentences': 3},
        {'name': 'Extended', 'target_char_min': 1000, 'target_char_max': 1300, 'target_sentences': 4},
    ]
    
    sample_text = demo_texts["Technical Document"]
    
    for config in configurations:
        name = config.pop('name')
        print(f"\n{name} Configuration: {config}")
        
        config_segmenter = DocumentSegmenter(**config)
        segments = config_segmenter.segment_document(sample_text)
        report = config_segmenter.get_segmentation_report()
        
        print(f"  Segments created: {len(segments)}")
        print(f"  Avg char length: {report['summary']['avg_char_length']:.1f}")
        print(f"  Avg sentence count: {report['summary']['avg_sentence_count']:.1f}")
        print(f"  Quality score: {report['quality_indicators']['overall_quality_score']:.3f}")
    
    print(f"\n{'='*60}")
    print("Demo completed successfully!")
    print("Check the logs for detailed segmentation metrics.")
    print('='*60)


if __name__ == "__main__":
    main()