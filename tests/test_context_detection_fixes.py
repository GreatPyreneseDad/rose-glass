#!/usr/bin/env python3
"""
Test Context Detection Fixes
============================

Validates that the reported issues are fixed:
1. Context type detection (crisis/mission/standard)
2. Emotional signature calibration
3. Trauma-informed flag activation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "emotionally-informed-rag" / "agent"))

from mcp_server_production import ProductionLegalRAGServer


def test_context_detection():
    """Test that context type detection works correctly"""

    server = ProductionLegalRAGServer()

    test_cases = [
        # (query, expected_context, expected_trauma, min_q)
        ("motion vacate restraining order Fardell custody Maple", "crisis", True, 0.0),
        ("PPO personal protection order Fardell", "crisis", True, 0.0),
        ("Fardell MacGregor custody parenting time", "crisis", True, 0.0),
        ("Maple custody emergency hearing", "crisis", True, 0.4),
        ("analyze comprehensive legal precedent", "mission", False, 0.0),
        ("research statutory framework", "mission", False, 0.0),
        ("file motion to continue", "standard", False, 0.0),
    ]

    print("=" * 70)
    print("CONTEXT DETECTION TEST")
    print("=" * 70)

    all_passed = True

    for query, expected_context, expected_trauma, min_q in test_cases:
        result = server._analyze_emotional_context(query)

        context_match = result['context_type'] == expected_context
        trauma_match = result['trauma_informed'] == expected_trauma
        q_check = result['emotional_activation'] >= min_q

        passed = context_match and trauma_match and q_check

        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"\n{status}: {query}")
        print(f"  Expected: context={expected_context}, trauma={expected_trauma}, q>={min_q}")
        print(f"  Got:      context={result['context_type']}, "
              f"trauma={result['trauma_informed']}, q={result['emotional_activation']}")

        if not passed:
            all_passed = False
            if not context_match:
                print(f"  ERROR: Context mismatch!")
            if not trauma_match:
                print(f"  ERROR: Trauma flag mismatch!")
            if not q_check:
                print(f"  ERROR: q score too low!")

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


def test_emotional_calibration():
    """Test that emotional signatures are properly calibrated"""

    server = ProductionLegalRAGServer()

    test_cases = [
        # (description, text, expected_q_range, expected_rho_range)
        ("Bank statement (neutral)", "Checking account September 2025 balance $1,234.56", (0.0, 0.2), (0.0, 0.3)),
        ("Custody motion (elevated)", "Emergency motion for custody modification", (0.3, 0.6), (0.2, 0.5)),
        ("Threat email (high)", "I will harm you if you don't comply immediately", (0.6, 1.0), (0.0, 0.3)),
        ("Love letter (low)", "I cherish our memories together", (0.0, 0.2), (0.0, 0.2)),
        ("Legal brief (wisdom)", "Statutory analysis of constitutional precedent", (0.0, 0.3), (0.5, 1.0)),
    ]

    print("\n" + "=" * 70)
    print("EMOTIONAL SIGNATURE CALIBRATION TEST")
    print("=" * 70)

    all_passed = True

    for description, text, (q_min, q_max), (rho_min, rho_max) in test_cases:
        result = server._analyze_emotional_context(text)

        q = result['emotional_activation']
        rho = result['wisdom_depth']

        q_in_range = q_min <= q <= q_max
        rho_in_range = rho_min <= rho <= rho_max

        passed = q_in_range and rho_in_range

        status = "✓ PASS" if passed else "✗ FAIL"

        print(f"\n{status}: {description}")
        print(f"  Text: {text[:60]}...")
        print(f"  Expected: q ∈ [{q_min}, {q_max}], ρ ∈ [{rho_min}, {rho_max}]")
        print(f"  Got:      q = {q}, ρ = {rho}")

        if not passed:
            all_passed = False
            if not q_in_range:
                print(f"  ERROR: q out of range!")
            if not rho_in_range:
                print(f"  ERROR: ρ out of range!")

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL CALIBRATION TESTS PASSED")
    else:
        print("✗ SOME CALIBRATION TESTS FAILED")
    print("=" * 70)

    return all_passed


def test_trauma_triggers():
    """Test that trauma-informed flag activates correctly"""

    server = ProductionLegalRAGServer()

    trauma_queries = [
        "custody emergency",
        "protection order violation",
        "restraining order",
        "domestic violence report",
        "child abuse investigation",
        "threat assessment",
    ]

    non_trauma_queries = [
        "file motion to continue",
        "discovery request documents",
        "bank account statement",
        "research legal framework",
    ]

    print("\n" + "=" * 70)
    print("TRAUMA-INFORMED FLAG TEST")
    print("=" * 70)

    all_passed = True

    print("\nQueries that SHOULD trigger trauma flag:")
    for query in trauma_queries:
        result = server._analyze_emotional_context(query)
        passed = result['trauma_informed']
        status = "✓" if passed else "✗ FAIL"
        print(f"  {status} {query}: trauma={result['trauma_informed']}")
        if not passed:
            all_passed = False

    print("\nQueries that should NOT trigger trauma flag:")
    for query in non_trauma_queries:
        result = server._analyze_emotional_context(query)
        passed = not result['trauma_informed']
        status = "✓" if passed else "✗ FAIL"
        print(f"  {status} {query}: trauma={result['trauma_informed']}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TRAUMA FLAG TESTS PASSED")
    else:
        print("✗ SOME TRAUMA FLAG TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║  Rose Glass Context Detection Fix Validation" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝\n")

    results = []

    # Run tests
    results.append(("Context Detection", test_context_detection()))
    results.append(("Emotional Calibration", test_emotional_calibration()))
    results.append(("Trauma Flag", test_trauma_triggers()))

    # Summary
    print("\n" + "╔" + "=" * 68 + "╗")
    print("║  TEST SUMMARY" + " " * 54 + "║")
    print("╚" + "=" * 68 + "╝\n")

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(result[1] for result in results)

    print("\n" + ("=" * 70))
    if all_passed:
        print("✅ ALL TESTS PASSED - Fixes validated!")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    print("=" * 70 + "\n")

    sys.exit(0 if all_passed else 1)
