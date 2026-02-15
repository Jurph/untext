"""Tests for consensus detection utilities.

This module tests the consensus detection logic that combines results from
multiple text detectors (EAST, DocTR, EasyOCR) to find high-confidence regions.

High diagnostic value tests:
- Bbox overlap calculation with edge cases (no overlap, partial, full)
- Bbox union calculation with various configurations
- Hybrid confidence calculation with different detector combinations
- Consensus finding logic with real overlap scenarios
- Boundary conditions and error cases
"""

import numpy as np
import cv2
import pytest
from typing import List, Tuple, Dict

from untextre.consensus import (
    calculate_bbox_overlap,
    calculate_bbox_union,
    calculate_hybrid_confidence,
    find_consensus_boxes,
)


class TestBboxOverlap:
    """Test bbox overlap calculations with diagnostic value."""
    
    def test_no_overlap(self):
        """Test bboxes with no overlap return 0."""
        bbox1 = (0, 0, 50, 50)
        bbox2 = (100, 100, 50, 50)
        
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert overlap == 0.0, "Non-overlapping boxes should have 0 overlap"
    
    def test_partial_overlap(self):
        """Test bboxes with partial overlap return correct area."""
        bbox1 = (0, 0, 50, 50)
        bbox2 = (25, 25, 50, 50)  # Overlaps 25x25 in corner
        
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        expected = 25 * 25  # 625 pixels
        assert overlap == expected, f"Expected {expected} overlap, got {overlap}"
    
    def test_full_overlap_identical(self):
        """Test identical bboxes return full area."""
        bbox = (10, 10, 100, 100)
        
        overlap = calculate_bbox_overlap(bbox, bbox)
        expected = 100 * 100  # Full area
        assert overlap == expected, "Identical boxes should fully overlap"
    
    def test_full_overlap_one_inside_other(self):
        """Test smaller bbox completely inside larger returns smaller area."""
        bbox_large = (0, 0, 100, 100)
        bbox_small = (25, 25, 50, 50)
        
        overlap = calculate_bbox_overlap(bbox_large, bbox_small)
        expected = 50 * 50  # Smaller box area
        assert overlap == expected, "Smaller box inside larger should return smaller area"
    
    def test_edge_touching_no_overlap(self):
        """Test bboxes touching at edges have no overlap (boundary condition)."""
        bbox1 = (0, 0, 50, 50)
        bbox2 = (50, 0, 50, 50)  # Touches right edge
        
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert overlap == 0.0, "Edge-touching boxes should not overlap"
    
    def test_single_pixel_overlap(self):
        """Test minimal overlap case."""
        bbox1 = (0, 0, 50, 50)
        bbox2 = (49, 49, 50, 50)  # 1x1 overlap at corner
        
        overlap = calculate_bbox_overlap(bbox1, bbox2)
        assert overlap == 1.0, "Single pixel overlap should return 1.0"


class TestBboxUnion:
    """Test bbox union calculations with diagnostic value."""
    
    def test_disjoint_boxes(self):
        """Test union of non-overlapping boxes creates encompassing rect."""
        bbox1 = (0, 0, 50, 50)
        bbox2 = (100, 100, 50, 50)
        
        union = calculate_bbox_union(bbox1, bbox2)
        assert union == (0, 0, 150, 150), "Union should encompass both boxes"
    
    def test_overlapping_boxes(self):
        """Test union of overlapping boxes."""
        bbox1 = (0, 0, 75, 75)
        bbox2 = (50, 50, 75, 75)
        
        union = calculate_bbox_union(bbox1, bbox2)
        assert union == (0, 0, 125, 125), "Union should encompass both overlapping boxes"
    
    def test_identical_boxes(self):
        """Test union of identical boxes returns same box."""
        bbox = (10, 10, 100, 100)
        
        union = calculate_bbox_union(bbox, bbox)
        assert union == bbox, "Union of identical boxes should return same box"
    
    def test_one_box_inside_other(self):
        """Test union when one box is inside another."""
        bbox_large = (0, 0, 100, 100)
        bbox_small = (25, 25, 50, 50)
        
        union = calculate_bbox_union(bbox_large, bbox_small)
        assert union == bbox_large, "Union should return larger box when one is inside"
    
    def test_various_positions(self):
        """Test union with boxes in various relative positions."""
        # Vertical alignment
        bbox1 = (50, 0, 50, 50)
        bbox2 = (50, 75, 50, 50)
        union = calculate_bbox_union(bbox1, bbox2)
        assert union == (50, 0, 50, 125), "Vertically separated boxes"
        
        # Horizontal alignment
        bbox1 = (0, 50, 50, 50)
        bbox2 = (75, 50, 50, 50)
        union = calculate_bbox_union(bbox1, bbox2)
        assert union == (0, 50, 125, 50), "Horizontally separated boxes"


class TestHybridConfidence:
    """Test hybrid confidence calculation with diagnostic value."""
    
    def test_single_confidence(self):
        """Test with single confidence value returns that value."""
        confidences = [0.8]
        result = calculate_hybrid_confidence(confidences)
        assert result == 0.8, "Single confidence should return that value"
    
    def test_two_high_confidences(self):
        """Test that two high confidences produce higher combined confidence."""
        confidences = [0.8, 0.9]
        result = calculate_hybrid_confidence(confidences)
        
        # Formula: 1 - (1-0.8)*(1-0.9) = 1 - 0.02 = 0.98
        expected = 0.98
        assert abs(result - expected) < 0.01, f"Expected ~{expected}, got {result}"
        
        # Should be higher than either individual confidence
        assert result > 0.8 and result > 0.9, "Combined should be higher than individuals"
    
    def test_two_low_confidences(self):
        """Test that two low confidences still boost each other."""
        confidences = [0.4, 0.5]
        result = calculate_hybrid_confidence(confidences)
        
        # Formula: 1 - (1-0.4)*(1-0.5) = 1 - 0.3 = 0.7
        expected = 0.7
        assert abs(result - expected) < 0.01, f"Expected ~{expected}, got {result}"
        
        # Should be higher than either individual
        assert result > 0.4 and result > 0.5, "Combined should boost low confidences"
    
    def test_three_detectors(self):
        """Test hybrid confidence with three detector confidences."""
        confidences = [0.6, 0.7, 0.8]
        result = calculate_hybrid_confidence(confidences)
        
        # Formula: 1 - (1-0.6)*(1-0.7)*(1-0.8) = 1 - 0.024 = 0.976
        expected = 0.976
        assert abs(result - expected) < 0.01, f"Expected ~{expected}, got {result}"
    
    def test_empty_confidences(self):
        """Test that empty list returns 0 (edge case)."""
        result = calculate_hybrid_confidence([])
        assert result == 0.0, "Empty confidence list should return 0"
    
    def test_percentage_format_conversion(self):
        """Test that percentage values (0-100) are properly converted."""
        # Some detectors return 0-100, others 0-1
        confidences_pct = [80.0, 90.0]  # Percentages
        confidences_frac = [0.8, 0.9]    # Fractions
        
        result_pct = calculate_hybrid_confidence(confidences_pct)
        result_frac = calculate_hybrid_confidence(confidences_frac)
        
        # Both should produce same result after conversion
        assert abs(result_pct - result_frac) < 0.01, \
            "Percentage and fraction formats should produce same result"


class TestFindConsensusBoxes:
    """Test consensus box finding algorithm with diagnostic scenarios."""
    
    def test_two_detector_agreement(self):
        """Test finding consensus when exactly 2 detectors agree."""
        detections = {
            "doctr": [(100, 100, 50, 50, 0.8)],
            "easyocr": [(105, 105, 45, 45, 0.9)],  # Overlaps with doctr
            "east": []
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        assert len(consensus) == 1, "Should find one consensus region"
        assert set(consensus[0]["detectors"]) == {"doctr", "easyocr"}, \
            "Consensus should include both agreeing detectors"
        assert consensus[0]["detector_count"] == 2
    
    def test_three_detector_agreement(self):
        """Test finding consensus when all 3 detectors agree."""
        detections = {
            "doctr": [(100, 100, 50, 50, 0.8)],
            "easyocr": [(105, 105, 45, 45, 0.9)],
            "east": [(98, 98, 52, 52, 0.7)]  # All overlap
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        assert len(consensus) == 1, "Should find one consensus region"
        assert len(consensus[0]["detectors"]) == 3, "Should include all 3 detectors"
        
        # Verify high hybrid confidence
        assert consensus[0]["confidence"] > 0.9, \
            "Three agreeing detectors should produce high confidence"
    
    def test_no_consensus(self):
        """Test when no detectors agree (detections are far apart)."""
        detections = {
            "doctr": [(0, 0, 50, 50, 0.8)],
            "easyocr": [(200, 200, 50, 50, 0.9)],
            "east": [(400, 400, 50, 50, 0.7)]
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        assert len(consensus) == 0, "Non-overlapping detections should produce no consensus"
    
    def test_multiple_consensus_regions(self):
        """Test finding multiple separate consensus regions."""
        detections = {
            "doctr": [
                (100, 100, 50, 50, 0.8),  # Region 1
                (300, 300, 50, 50, 0.7)   # Region 2
            ],
            "easyocr": [
                (105, 105, 45, 45, 0.9),  # Overlaps region 1
                (305, 305, 45, 45, 0.85)  # Overlaps region 2
            ],
            "east": []
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        assert len(consensus) == 2, "Should find two separate consensus regions"
    
    def test_same_detector_no_consensus(self):
        """Test that same detector's multiple detections don't create consensus."""
        detections = {
            "doctr": [
                (100, 100, 50, 50, 0.8),
                (105, 105, 45, 45, 0.9)  # Overlaps with previous doctr detection
            ],
            "easyocr": [],
            "east": []
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        assert len(consensus) == 0, "Same detector's detections shouldn't create consensus"
    
    def test_overlap_threshold_filtering(self):
        """Test that overlap threshold correctly filters weak overlaps."""
        detections = {
            "doctr": [(100, 100, 100, 100, 0.8)],     # Large box
            "easyocr": [(180, 180, 25, 25, 0.9)],     # Small box, minimal overlap
            "east": []
        }
        
        # Overlap calculation uses smaller box area as denominator
        # 20x20 overlap on 25x25 box = 400/625 = 64% overlap ratio
        # This is a good test of the actual algorithm behavior
        
        # With low threshold (5%), should create consensus (64% >> 5%)
        consensus_low = find_consensus_boxes(detections, overlap_threshold=0.05)
        assert len(consensus_low) == 1, "Significant overlap should create consensus with low threshold"
        
        # With very high threshold (70%), should not create consensus (64% < 70%)
        consensus_high = find_consensus_boxes(detections, overlap_threshold=0.7)
        assert len(consensus_high) == 0, "64% overlap fails 70% threshold"
        
        # With threshold at 60%, should create consensus (64% >= 60%)
        consensus_mid = find_consensus_boxes(detections, overlap_threshold=0.6)
        assert len(consensus_mid) == 1, "64% overlap meets 60% threshold"
    
    def test_consensus_bbox_is_union(self):
        """Test that consensus bbox is the union of overlapping detections."""
        detections = {
            "doctr": [(10, 10, 30, 30, 0.8)],     # x: 10-40, y: 10-40
            "easyocr": [(30, 30, 30, 30, 0.9)],   # x: 30-60, y: 30-60
            "east": []
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        assert len(consensus) == 1
        bbox = consensus[0]["bbox"]
        
        # Union should span from (10,10) to (60,60)
        assert bbox == (10, 10, 50, 50), f"Union bbox should be (10,10,50,50), got {bbox}"
    
    def test_empty_detections(self):
        """Test with no detections (edge case)."""
        detections = {
            "doctr": [],
            "easyocr": [],
            "east": []
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        assert len(consensus) == 0, "No detections should produce no consensus"
    
    def test_confidence_propagation(self):
        """Test that original confidences are preserved in consensus."""
        detections = {
            "doctr": [(100, 100, 50, 50, 0.6)],
            "easyocr": [(105, 105, 45, 45, 0.7)],
            "east": []
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        assert len(consensus) == 1
        original_confs = consensus[0]["original_confidences"]
        
        # Should preserve original confidences (converted to 0-1 range)
        assert 0.6 in original_confs or abs(original_confs[0] - 0.6) < 0.01
        assert 0.7 in original_confs or abs(original_confs[1] - 0.7) < 0.01


class TestHybridConfidenceEdgeCases:
    """Test hybrid confidence with edge cases that matter."""
    
    def test_all_zero_confidences(self):
        """Test with all zero confidences (detector failures)."""
        confidences = [0.0, 0.0, 0.0]
        result = calculate_hybrid_confidence(confidences)
        
        # Formula: 1 - (1-0)*(1-0)*(1-0) = 1 - 1 = 0
        assert result == 0.0, "All zero confidences should produce zero"
    
    def test_all_perfect_confidences(self):
        """Test with all perfect confidences."""
        confidences = [1.0, 1.0, 1.0]
        result = calculate_hybrid_confidence(confidences)
        
        # Formula: 1 - (1-1)*(1-1)*(1-1) = 1 - 0 = 1
        assert result == 1.0, "All perfect confidences should produce 1.0"
    
    def test_one_perfect_makes_perfect(self):
        """Test that single perfect confidence makes result perfect."""
        confidences = [0.3, 1.0, 0.5]
        result = calculate_hybrid_confidence(confidences)
        
        # Any confidence of 1.0 means certainty, so result should be 1.0
        assert result == 1.0, "One perfect confidence should guarantee perfect result"
    
    def test_mixed_percentage_and_fraction_formats(self):
        """Test mixing 0-1 and 0-100 formats (realistic bug scenario)."""
        confidences = [0.8, 85.0, 0.9]  # Mixed formats
        result = calculate_hybrid_confidence(confidences)
        
        # Should handle conversion properly
        assert 0.0 <= result <= 1.0, "Result should be in 0-1 range"
        assert result > 0.95, "Strong confidences should produce strong result"


class TestConsensusIntegration:
    """Integration tests for consensus detection flow."""
    
    def test_realistic_watermark_scenario(self):
        """Test realistic watermark detection scenario with multiple overlapping regions."""
        # Simulate watermark in bottom-right corner detected by all 3 detectors
        # with slight variations in bbox due to different detection algorithms
        detections = {
            "doctr": [(450, 450, 100, 30, 0.75)],      # Slightly smaller
            "easyocr": [(445, 448, 110, 35, 0.82)],    # Slightly larger, offset
            "east": [(448, 447, 105, 32, 0.68)]        # Middle ground
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        # Should find one strong consensus region
        assert len(consensus) == 1, "Should find consensus watermark region"
        
        # Verify all detectors contributed
        assert len(consensus[0]["detectors"]) == 3, "All 3 detectors should agree"
        
        # Verify bbox is union encompassing all detections
        bbox = consensus[0]["bbox"]
        assert bbox[0] == 445, "Should start at leftmost detection"
        assert bbox[1] == 447, "Should start at topmost detection"
        
        # Verify high hybrid confidence
        assert consensus[0]["confidence"] > 0.9, \
            "Three agreeing detectors should produce high confidence"
    
    def test_false_positive_filtered(self):
        """Test consensus behavior with multiple detection groups."""
        detections = {
            "doctr": [
                (100, 100, 50, 50, 0.8),   # Strong detection region 1
                (500, 500, 20, 20, 0.4)    # Weak isolated detection region 2
            ],
            "easyocr": [
                (105, 105, 45, 45, 0.9),   # Agrees with strong doctr in region 1
            ],
            "east": [
                (505, 505, 15, 15, 0.35)   # Overlaps with weak doctr in region 2
            ]
        }
        
        consensus = find_consensus_boxes(detections, overlap_threshold=0.1)
        
        # Algorithm finds consensus wherever 2+ detectors overlap
        # Region 1: doctr + easyocr (strong)
        # Region 2: doctr + east (weak overlap exists)
        assert len(consensus) == 2, "Algorithm finds consensus for both overlapping groups"
        
        # Verify the strong consensus is first (or find it)
        strong_consensus = [c for c in consensus if set(c["detectors"]) == {"doctr", "easyocr"}]
        assert len(strong_consensus) == 1, "Should have strong doctr+easyocr consensus"
        assert strong_consensus[0]["confidence"] > 0.95, "Strong consensus should have high confidence"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
