#!/usr/bin/env python3
"""
Script principal pour exécuter le pipeline HygieSync complet
Usage: python run_pipeline.py [command] [args]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_test():
    """Exécute les tests du pipeline"""
    from test_pipeline import main
    return main()


def cmd_probe(video_path: str):
    """Vérifie la détection des landmarks"""
    from src.probe_landmarks import probe_landmarks
    return probe_landmarks(video_path)


def cmd_prepare(video_path: str, out_dir: str):
    """Prépare le dataset"""
    from src.prepare_dataset import prepare_dataset
    return prepare_dataset(video_path, out_dir)


def cmd_train(ds_dir: str, out_dir: str):
    """Entraîne le modèle"""
    from src.train import train
    return train(ds_dir, out_dir)


def cmd_infer(ckpt: str, template: str, audio: str, output: str):
    """Génère une vidéo lip-sync"""
    from src.infer import infer
    return infer(ckpt, template, audio, output)


def cmd_full(train_video: str, template_video: str, audio: str, output: str):
    """Pipeline complet: prepare -> train -> infer"""
    ds_dir = "data/ds"
    runs_dir = "runs/hygie"
    
    print("\n" + "=" * 60)
    print("  HYGIE-SYNC FULL PIPELINE")
    print("=" * 60)
    
    print("\n[1/4] Probing landmarks...")
    if not cmd_probe(train_video):
        print("ERROR: Landmark detection failed")
        return False
    
    print("\n[2/4] Preparing dataset...")
    cmd_prepare(train_video, ds_dir)
    
    print("\n[3/4] Training model...")
    cmd_train(ds_dir, runs_dir)
    
    print("\n[4/4] Generating video...")
    ckpt = os.path.join(runs_dir, "ckpt_best.pt")
    cmd_infer(ckpt, template_video, audio, output)
    
    print("\n" + "=" * 60)
    print(f"  DONE! Output: {output}")
    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="HygieSync Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    sub_test = subparsers.add_parser("test", help="Run pipeline tests")
    
    sub_probe = subparsers.add_parser("probe", help="Check landmark detection")
    sub_probe.add_argument("video", help="Video path")
    
    sub_prepare = subparsers.add_parser("prepare", help="Prepare dataset")
    sub_prepare.add_argument("video", help="Training video path")
    sub_prepare.add_argument("--out", default="data/ds", help="Output directory")
    
    sub_train = subparsers.add_parser("train", help="Train model")
    sub_train.add_argument("--ds", default="data/ds", help="Dataset directory")
    sub_train.add_argument("--out", default="runs/hygie", help="Output directory")
    
    sub_infer = subparsers.add_parser("infer", help="Generate lip-sync video")
    sub_infer.add_argument("--ckpt", default="runs/hygie/ckpt_best.pt", help="Checkpoint path")
    sub_infer.add_argument("--template", default="data/template_idle.mp4", help="Template video")
    sub_infer.add_argument("--audio", default="data/new_audio.wav", help="Audio file")
    sub_infer.add_argument("--output", default="out_sync.mp4", help="Output video")
    
    sub_full = subparsers.add_parser("full", help="Run full pipeline")
    sub_full.add_argument("--train-video", default="data/train_video.mp4", help="Training video")
    sub_full.add_argument("--template", default="data/template_idle.mp4", help="Template video")
    sub_full.add_argument("--audio", default="data/new_audio.wav", help="Audio file")
    sub_full.add_argument("--output", default="out_sync.mp4", help="Output video")
    
    args = parser.parse_args()
    
    if args.command == "test":
        return cmd_test()
    elif args.command == "probe":
        return cmd_probe(args.video)
    elif args.command == "prepare":
        return cmd_prepare(args.video, args.out)
    elif args.command == "train":
        return cmd_train(args.ds, args.out)
    elif args.command == "infer":
        return cmd_infer(args.ckpt, args.template, args.audio, args.output)
    elif args.command == "full":
        return cmd_full(args.train_video, args.template, args.audio, args.output)
    else:
        parser.print_help()
        print("\nQuick start:")
        print("  1. python run_pipeline.py test")
        print("  2. Place your videos in data/")
        print("  3. python run_pipeline.py full")


if __name__ == "__main__":
    main()
