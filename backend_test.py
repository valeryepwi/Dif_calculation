import requests
import json
import sys
from datetime import datetime

class SameMaterialBugTester:
    def __init__(self, base_url="https://diffusion-bond-tool.preview.emergentagent.com"):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.failed_tests = []

    def run_test(self, name, test_func):
        """Run a single test"""
        self.tests_run += 1
        print(f"\n🔍 {self.tests_run}. Testing {name}...")
        
        try:
            result = test_func()
            if result:
                self.tests_passed += 1
                print(f"✅ PASS - {name}")
                return True
            else:
                print(f"❌ FAIL - {name}")
                self.failed_tests.append(name)
                return False
        except Exception as e:
            print(f"❌ ERROR - {name}: {str(e)}")
            self.failed_tests.append(f"{name} (Exception: {str(e)})")
            return False

    def get_materials(self):
        """Get all materials and find MA2-1 and VT1-0 IDs"""
        try:
            response = requests.get(f"{self.base_url}/api/materials")
            if response.status_code != 200:
                print(f"Failed to get materials: {response.status_code}")
                return None, None, None
            
            materials = response.json()
            ma21_id = None
            vt10_id = None
            
            for mat in materials:
                if mat.get('grade') == 'MA2-1':
                    ma21_id = mat['id']
                elif mat.get('grade') == 'VT1-0':
                    vt10_id = mat['id']
            
            print(f"Found materials - MA2-1 ID: {ma21_id}, VT1-0 ID: {vt10_id}")
            return ma21_id, vt10_id, materials
            
        except Exception as e:
            print(f"Error getting materials: {e}")
            return None, None, None

    def test_same_material_calculation(self):
        """Test calculation with same material (MA2-1/MA2-1)"""
        ma21_id, vt10_id, _ = self.get_materials()
        if not ma21_id:
            print("MA2-1 material not found")
            return False
            
        payload = {
            "plate_material_id": ma21_id,
            "interlayer_material_id": ma21_id,  # Same material
            "plate_thickness_mm": 2.0,
            "plate_width_mm": 10.0,
            "plate_length_mm": 50.0,
            "interlayer_thickness_um": 20.0,
            "interface_params": {
                "diffusion_layer_thickness_um": 5.0,
                "quality_coefficient": 0.85,
                "kirkendall_porosity": 0.05,
                "intermetallic_thickness_um": 2.0,
                "intermetallic_E_GPa": 150.0,
                "intermetallic_strength_MPa": 200.0,
                "intermetallic_elongation_pct": 0.5
            },
            "name": "Same Material Test"
        }
        
        response = requests.post(f"{self.base_url}/api/calculate", json=payload)
        if response.status_code != 200:
            print(f"Calculation failed: {response.status_code} - {response.text}")
            return False
            
        result = response.json()
        self.same_material_result = result  # Store for other tests
        return True

    def test_same_material_notes(self):
        """Test that same-material calculation returns notes array with same_material warning"""
        if not hasattr(self, 'same_material_result'):
            return False
            
        notes = self.same_material_result.get('notes', [])
        if not notes:
            print("No notes found in same-material calculation")
            return False
            
        same_material_note = None
        for note in notes:
            if note.get('type') == 'same_material':
                same_material_note = note
                break
                
        if not same_material_note:
            print("No same_material note found in notes array")
            return False
            
        print(f"Found same_material note: {same_material_note.get('en', 'No English text')}")
        return True

    def test_same_material_no_intermetallic_zone(self):
        """Test that same-material stress distribution has NO intermetallic zone"""
        if not hasattr(self, 'same_material_result'):
            return False
            
        stress_dist = self.same_material_result.get('stress_distribution', [])
        if not stress_dist:
            print("No stress distribution data found")
            return False
            
        # Check for any intermetallic zones
        intermetallic_zones = [d for d in stress_dist if d.get('zone') == 'intermetallic']
        if intermetallic_zones:
            print(f"Found {len(intermetallic_zones)} intermetallic zone entries - should be 0!")
            return False
            
        print("No intermetallic zones found - correct for same material")
        return True

    def test_same_material_flat_stress(self):
        """Test that same-material stress distribution is approximately flat"""
        if not hasattr(self, 'same_material_result'):
            return False
            
        stress_dist = self.same_material_result.get('stress_distribution', [])
        if not stress_dist:
            return False
            
        stresses = [d.get('stress_mpa', 0) for d in stress_dist]
        if not stresses:
            return False
            
        min_stress = min(stresses)
        max_stress = max(stresses)
        stress_range = max_stress - min_stress
        avg_stress = sum(stresses) / len(stresses)
        
        # For same material, stress should be relatively flat (less than 5% variation)
        variation_percent = (stress_range / avg_stress) * 100 if avg_stress > 0 else 0
        
        print(f"Stress variation: {variation_percent:.2f}% (range: {min_stress:.2f}-{max_stress:.2f} MPa)")
        
        if variation_percent > 5:  # Allow small variation due to numerical effects
            print(f"Stress variation too high for same material: {variation_percent:.2f}%")
            return False
            
        return True

    def test_same_material_equal_uts(self):
        """Test that same-material all 5 models give approximately equal UTS values"""
        if not hasattr(self, 'same_material_result'):
            return False
            
        models = self.same_material_result.get('models', [])
        if len(models) != 5:
            print(f"Expected 5 models, got {len(models)}")
            return False
            
        uts_values = [m.get('uts_mpa', 0) for m in models]
        model_names = [m.get('name', 'unknown') for m in models]
        
        if not uts_values:
            return False
            
        min_uts = min(uts_values)
        max_uts = max(uts_values)
        avg_uts = sum(uts_values) / len(uts_values)
        uts_range = max_uts - min_uts
        
        # For same material, all models should give similar results (within 10% variation)
        variation_percent = (uts_range / avg_uts) * 100 if avg_uts > 0 else 0
        
        print(f"UTS variation across models: {variation_percent:.2f}%")
        for name, uts in zip(model_names, uts_values):
            print(f"  {name}: {uts:.2f} MPa")
            
        if variation_percent > 10:
            print(f"UTS variation too high for same material: {variation_percent:.2f}%")
            return False
            
        return True

    def test_different_material_calculation(self):
        """Test calculation with different materials (MA2-1/VT1-0)"""
        ma21_id, vt10_id, _ = self.get_materials()
        if not ma21_id or not vt10_id:
            print("Required materials not found")
            return False
            
        payload = {
            "plate_material_id": ma21_id,
            "interlayer_material_id": vt10_id,  # Different material
            "plate_thickness_mm": 2.0,
            "plate_width_mm": 10.0,
            "plate_length_mm": 50.0,
            "interlayer_thickness_um": 20.0,
            "interface_params": {
                "diffusion_layer_thickness_um": 5.0,
                "quality_coefficient": 0.85,
                "kirkendall_porosity": 0.05,
                "intermetallic_thickness_um": 2.0,
                "intermetallic_E_GPa": 150.0,
                "intermetallic_strength_MPa": 200.0,
                "intermetallic_elongation_pct": 0.5
            },
            "name": "Different Material Test"
        }
        
        response = requests.post(f"{self.base_url}/api/calculate", json=payload)
        if response.status_code != 200:
            print(f"Calculation failed: {response.status_code} - {response.text}")
            return False
            
        result = response.json()
        self.different_material_result = result  # Store for other tests
        return True

    def test_different_material_has_intermetallic(self):
        """Test that different-material calculation HAS intermetallic zones"""
        if not hasattr(self, 'different_material_result'):
            return False
            
        stress_dist = self.different_material_result.get('stress_distribution', [])
        if not stress_dist:
            return False
            
        # Check for intermetallic zones
        intermetallic_zones = [d for d in stress_dist if d.get('zone') == 'intermetallic']
        if not intermetallic_zones:
            print("No intermetallic zones found - should have some for different materials!")
            return False
            
        print(f"Found {len(intermetallic_zones)} intermetallic zone entries - correct for different materials")
        return True

    def test_different_material_varying_stress(self):
        """Test that different-material calculation has varying stress (not flat)"""
        if not hasattr(self, 'different_material_result'):
            return False
            
        stress_dist = self.different_material_result.get('stress_distribution', [])
        if not stress_dist:
            return False
            
        stresses = [d.get('stress_mpa', 0) for d in stress_dist]
        if not stresses:
            return False
            
        min_stress = min(stresses)
        max_stress = max(stresses)
        stress_range = max_stress - min_stress
        avg_stress = sum(stresses) / len(stresses)
        
        # For different materials, stress should vary significantly (>5%)
        variation_percent = (stress_range / avg_stress) * 100 if avg_stress > 0 else 0
        
        print(f"Stress variation: {variation_percent:.2f}% (range: {min_stress:.2f}-{max_stress:.2f} MPa)")
        
        if variation_percent <= 5:
            print(f"Stress variation too low for different materials: {variation_percent:.2f}%")
            return False
            
        return True

    def test_different_material_empty_notes(self):
        """Test that different-material calculation returns empty notes array"""
        if not hasattr(self, 'different_material_result'):
            return False
            
        notes = self.different_material_result.get('notes', [])
        if notes:
            print(f"Found {len(notes)} notes - should be empty for different materials")
            print("Notes:", [note.get('type', 'unknown') for note in notes])
            return False
            
        print("Notes array is empty - correct for different materials")
        return True

    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Starting Same-Material Bug Fix Tests")
        print(f"Testing against: {self.base_url}")
        print("=" * 60)
        
        # Backend tests
        self.run_test("Backend: Same-material calculation (MA2-1/MA2-1)", self.test_same_material_calculation)
        self.run_test("Backend: Same-material returns notes with same_material warning", self.test_same_material_notes)
        self.run_test("Backend: Same-material has NO intermetallic zone", self.test_same_material_no_intermetallic_zone)
        self.run_test("Backend: Same-material stress is flat (all values equal)", self.test_same_material_flat_stress)
        self.run_test("Backend: Same-material all 5 models give equal UTS", self.test_same_material_equal_uts)
        
        self.run_test("Backend: Different-material calculation (MA2-1/VT1-0)", self.test_different_material_calculation)
        self.run_test("Backend: Different-material HAS intermetallic zone", self.test_different_material_has_intermetallic)
        self.run_test("Backend: Different-material has varying stress", self.test_different_material_varying_stress)
        self.run_test("Backend: Different-material returns empty notes", self.test_different_material_empty_notes)

        # Print results
        print("\n" + "=" * 60)
        print(f"📊 Test Results: {self.tests_passed}/{self.tests_run} passed")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.failed_tests:
            print(f"\n❌ Failed Tests ({len(self.failed_tests)}):")
            for test in self.failed_tests:
                print(f"  • {test}")
                
        return self.tests_passed == self.tests_run

def main():
    tester = SameMaterialBugTester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())