#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command-Line Interface for Policy Analysis System
================================================
Provides command-line access to the policy analysis and feasibility scoring system
with configurable parameters for parallel processing, device selection, and output control.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser with all supported flags."""
    
    parser = argparse.ArgumentParser(
        description="Policy Analysis System - Feasibility Scoring and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --input ./documents --outdir ./results
  python cli.py --input ./documents --workers 8 --device cuda --precision float32
  python cli.py --input ./documents --topk 10 --umbral 0.75 --max-segmentos 1000
        """
    )
    
    # Input/Output paths
    parser.add_argument(
        '--input',
        type=str,
        default='.',
        help='Input directory path containing documents to analyze (default: current directory)'
    )
    
    parser.add_argument(
        '--outdir',
        type=str,
        default='output',
        help='Output directory path for results (default: "output")'
    )
    
    # Parallel processing configuration
    parser.add_argument(
        '--workers',
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help=f'Number of parallel workers for processing (default: {min(os.cpu_count() or 1, 8)})'
    )
    
    # Device selection for computation
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'cuda:0', 'cuda:1', 'mps'],
        help='Computation device selection (default: auto-detect)'
    )
    
    # Numerical precision settings
    parser.add_argument(
        '--precision',
        type=str,
        default='float32',
        choices=['float16', 'float32', 'float64'],
        help='Numerical precision for calculations (default: float32)'
    )
    
    # Top-k search results
    parser.add_argument(
        '--topk',
        type=int,
        default=10,
        help='Number of top-k search results to return (default: 10)'
    )
    
    # Threshold values
    parser.add_argument(
        '--umbral',
        type=float,
        default=0.5,
        help='Threshold value for similarity/confidence filtering (default: 0.5)'
    )
    
    # Maximum segments limit
    parser.add_argument(
        '--max-segmentos',
        type=int,
        default=1000,
        help='Maximum number of text segments to process (default: 1000)'
    )
    
    # Processing mode selection
    parser.add_argument(
        '--mode',
        type=str,
        default='feasibility',
        choices=['feasibility', 'decatalogo', 'embedding', 'demo'],
        help='Processing mode to execute (default: feasibility)'
    )
    
    # Additional options
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output for debugging'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON configuration file (overrides command-line options)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configuration without executing processing'
    )
    
    return parser


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    try:
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        sys.exit(1)


def validate_args(args: argparse.Namespace) -> None:
    """Validate and adjust parsed arguments."""
    
    # Validate input path exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path '{args.input}' does not exist")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(args.outdir)
    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating output directory '{args.outdir}': {e}")
        sys.exit(1)
    
    # Validate workers count
    if args.workers < 1:
        print("Error: Workers count must be at least 1")
        sys.exit(1)
    
    # Validate topk value
    if args.topk < 1:
        print("Error: topk value must be at least 1")
        sys.exit(1)
    
    # Validate umbral range
    if not 0.0 <= args.umbral <= 1.0:
        print("Error: umbral value must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Validate max_segmentos
    if args.max_segmentos < 1:
        print("Error: max-segmentos value must be at least 1")
        sys.exit(1)


def get_device_config(device_arg: str) -> str:
    """Determine the optimal device configuration."""
    if device_arg == 'auto':
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        except ImportError:
            return 'cpu'
    else:
        return device_arg


def setup_logging(verbose: bool = False):
    """Setup logging configuration based on verbosity level."""
    import logging
    
    level = logging.DEBUG if verbose else logging.INFO
    format_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('policy_analysis.log', encoding='utf-8')
        ]
    )


def run_feasibility_mode(args: argparse.Namespace) -> int:
    """Execute feasibility scoring mode."""
    try:
        from feasibility_scorer import FeasibilityScorer
        
        print(f"Running feasibility analysis...")
        print(f"Input directory: {args.input}")
        print(f"Output directory: {args.outdir}")
        print(f"Workers: {args.workers}")
        print(f"Device: {args.device}")
        print(f"Precision: {args.precision}")
        print(f"Top-k: {args.topk}")
        print(f"Umbral: {args.umbral}")
        print(f"Max segments: {args.max_segmentos}")
        
        # Initialize scorer with CLI parameters
        scorer = FeasibilityScorer(
            enable_parallel=args.workers > 1,
            n_jobs=args.workers,
            backend='loky'
        )
        
        # Process input directory for text files
        input_path = Path(args.input)
        text_files = []
        
        for ext in ['*.txt', '*.md', '*.pdf']:
            text_files.extend(input_path.glob(ext))
        
        if not text_files:
            print(f"No text files found in {args.input}")
            return 1
        
        print(f"Found {len(text_files)} files to process")
        
        # Read and process files
        indicators = []
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split content into segments up to max_segmentos
                    segments = content.split('\n')
                    segments = [s.strip() for s in segments if s.strip()]
                    segments = segments[:args.max_segmentos]
                    indicators.extend(segments)
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        if not indicators:
            print("No content found to analyze")
            return 1
        
        print(f"Analyzing {len(indicators)} indicators...")
        
        # Score indicators using CLI parameters
        if args.workers > 1:
            results = scorer.batch_score(
                indicators[:args.max_segmentos],
                compare_backends=args.verbose
            )
        else:
            results = [scorer.calculate_feasibility_score(ind) for ind in indicators[:args.max_segmentos]]
        
        # Filter results by umbral threshold
        filtered_results = [
            (ind, result) for ind, result in zip(indicators, results)
            if result.feasibility_score >= args.umbral
        ]
        
        # Sort by score and take top-k
        filtered_results.sort(key=lambda x: x[1].feasibility_score, reverse=True)
        top_results = filtered_results[:args.topk]
        
        # Generate report
        output_file = Path(args.outdir) / 'feasibility_report.json'
        report_data = {
            'config': {
                'input': args.input,
                'workers': args.workers,
                'device': args.device,
                'precision': args.precision,
                'topk': args.topk,
                'umbral': args.umbral,
                'max_segmentos': args.max_segmentos
            },
            'summary': {
                'total_indicators': len(indicators),
                'processed_indicators': min(len(indicators), args.max_segmentos),
                'passed_threshold': len(filtered_results),
                'top_k_results': len(top_results)
            },
            'results': [
                {
                    'text': text[:200] + ('...' if len(text) > 200 else ''),
                    'score': result.feasibility_score,
                    'quality_tier': result.quality_tier,
                    'components': [c.value for c in result.components_detected],
                    'quantitative_baseline': result.has_quantitative_baseline,
                    'quantitative_target': result.has_quantitative_target
                }
                for text, result in top_results
            ]
        }
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Analysis complete. Results saved to {output_file}")
        print(f"Top {len(top_results)} results (threshold >= {args.umbral}):")
        
        for i, (text, result) in enumerate(top_results[:5], 1):
            display_text = text[:100] + ('...' if len(text) > 100 else '')
            print(f"{i}. Score: {result.feasibility_score:.3f} | {result.quality_tier} | {display_text}")
        
        return 0
        
    except ImportError as e:
        print(f"Error: Required module not available: {e}")
        return 1
    except Exception as e:
        print(f"Error in feasibility mode: {e}")
        return 1


def run_embedding_mode(args: argparse.Namespace) -> int:
    """Execute embedding model mode."""
    try:
        from embedding_model import create_embedding_model
        
        print(f"Running embedding analysis...")
        
        # Get device configuration
        device = get_device_config(args.device)
        
        # Create embedding model with CLI parameters
        model = create_embedding_model(
            device=device,
            precision=args.precision,
            enable_cache=True
        )
        
        # Process input files
        input_path = Path(args.input)
        documents = []
        
        for file_path in input_path.glob('*.txt'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        documents.append({
                            'file': file_path.name,
                            'content': content[:args.max_segmentos]  # Limit content length
                        })
            except Exception as e:
                print(f"Warning: Could not read {file_path}: {e}")
        
        if not documents:
            print("No documents found to process")
            return 1
        
        print(f"Processing {len(documents)} documents...")
        
        # Generate embeddings
        texts = [doc['content'] for doc in documents]
        embeddings = model.encode(texts)
        
        print(f"Generated embeddings: {embeddings.shape}")
        
        # Save results
        output_file = Path(args.outdir) / 'embeddings.npy'
        metadata_file = Path(args.outdir) / 'embeddings_metadata.json'
        
        import numpy as np
        import json
        
        np.save(output_file, embeddings)
        
        metadata = {
            'config': {
                'device': device,
                'precision': args.precision,
                'max_segmentos': args.max_segmentos
            },
            'documents': [doc['file'] for doc in documents],
            'shape': list(embeddings.shape),
            'dtype': str(embeddings.dtype)
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Embeddings saved to {output_file}")
        print(f"Metadata saved to {metadata_file}")
        
        return 0
        
    except ImportError as e:
        print(f"Error: Required module not available: {e}")
        return 1
    except Exception as e:
        print(f"Error in embedding mode: {e}")
        return 1


def run_demo_mode(args: argparse.Namespace) -> int:
    """Execute demo mode."""
    try:
        # Pass CLI parameters through environment variables to maintain compatibility
        os.environ['CLI_WORKERS'] = str(args.workers)
        os.environ['CLI_DEVICE'] = args.device
        os.environ['CLI_OUTPUT_DIR'] = args.outdir
        
        print(f"Running demo mode with CLI configuration...")
        print(f"Workers: {args.workers}")
        print(f"Device: {args.device}")
        print(f"Output directory: {args.outdir}")
        
        # Import and run demo with environment configuration
        import demo
        demo.main()
        
        return 0
        
    except ImportError as e:
        print(f"Error: Demo module not available: {e}")
        return 1
    except Exception as e:
        print(f"Error in demo mode: {e}")
        return 1


def run_decatalogo_mode(args: argparse.Namespace) -> int:
    """Execute Decatalogo evaluation mode."""
    try:
        # Import with the CLI parameters passed as environment variables
        # This maintains backward compatibility with existing hardcoded values
        os.environ['CLI_WORKERS'] = str(args.workers)
        os.environ['CLI_DEVICE'] = args.device
        os.environ['CLI_PRECISION'] = args.precision
        os.environ['CLI_TOPK'] = str(args.topk)
        os.environ['CLI_UMBRAL'] = str(args.umbral)
        os.environ['CLI_MAX_SEGMENTOS'] = str(args.max_segmentos)
        os.environ['CLI_INPUT_DIR'] = args.input
        os.environ['CLI_OUTPUT_DIR'] = args.outdir
        
        print(f"Running Decatalogo evaluation...")
        print(f"Configuration passed via environment variables")
        
        # Import and run the evaluator
        from Decatalogo_evaluador import IndustrialDecatalogoEvaluatorFull
        
        evaluator = IndustrialDecatalogoEvaluatorFull()
        
        # Process input files
        input_path = Path(args.input)
        text_files = list(input_path.glob('*.txt'))
        
        if not text_files:
            print(f"No text files found in {args.input}")
            return 1
        
        print(f"Found {len(text_files)} files to evaluate")
        
        # Process each file
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Evaluate each of the 10 Decatalogo points
                for punto_id in range(1, 11):
                    result = evaluator.evaluar_punto_completo(content, punto_id)
                    
                    # Save individual results
                    output_file = Path(args.outdir) / f'decatalogo_punto_{punto_id}_{file_path.stem}.json'
                    
                    import json
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump({
                            'punto_id': result.punto_id,
                            'nombre_punto': result.nombre_punto,
                            'puntaje_agregado': result.puntaje_agregado_punto,
                            'evaluaciones_dimensiones': [
                                {
                                    'dimension': ed.dimension,
                                    'puntaje': ed.puntaje_dimension,
                                    'preguntas_evaluadas': len(ed.evaluaciones_preguntas)
                                }
                                for ed in result.evaluaciones_dimensiones
                            ]
                        }, f, indent=2, ensure_ascii=False)
                
                print(f"Processed {file_path.name}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        print(f"Decatalogo evaluation complete. Results in {args.outdir}")
        return 0
        
    except ImportError as e:
        print(f"Error: Decatalogo module not available: {e}")
        return 1
    except Exception as e:
        print(f"Error in decatalogo mode: {e}")
        return 1


def main():
    """Main entry point for the CLI application."""
    
    # Create and parse arguments
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration file if provided
    if args.config:
        config = load_config_file(args.config)
        # Override command line args with config file values
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    validate_args(args)
    
    # Show configuration and exit if dry-run
    if args.dry_run:
        print("Configuration (dry-run mode):")
        print(f"  Input directory: {args.input}")
        print(f"  Output directory: {args.outdir}")
        print(f"  Workers: {args.workers}")
        print(f"  Device: {get_device_config(args.device)}")
        print(f"  Precision: {args.precision}")
        print(f"  Top-k: {args.topk}")
        print(f"  Umbral: {args.umbral}")
        print(f"  Max segments: {args.max_segmentos}")
        print(f"  Mode: {args.mode}")
        print(f"  Verbose: {args.verbose}")
        return 0
    
    # Execute the selected mode
    if args.mode == 'feasibility':
        return run_feasibility_mode(args)
    elif args.mode == 'embedding':
        return run_embedding_mode(args)
    elif args.mode == 'demo':
        return run_demo_mode(args)
    elif args.mode == 'decatalogo':
        return run_decatalogo_mode(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())