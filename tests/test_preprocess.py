"""
Unit tests for preprocess.py module.
"""
import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocess import (
    extract_district_city,
    target_encode_with_smoothing,
    target_encode_with_oof,
    apply_target_encoding,
    preprocess_data,
    save_encodings,
    load_encodings
)


class TestExtractDistrictCity:
    """Tests for extract_district_city function."""

    def test_valid_address(self):
        """Test extracting district and city from valid address."""
        df = pd.DataFrame({
            'Address': ['123 Nguyen Trai, Go Vap, Ho Chi Minh']
        })
        result = extract_district_city(df)
        
        assert 'City' in result.columns
        assert 'District' in result.columns
        assert result['City'].iloc[0] == 'Ho Chi Minh'
        assert result['District'].iloc[0] == 'Go Vap'

    def test_address_without_comma(self):
        """Test address with single part defaults to Unknown."""
        df = pd.DataFrame({
            'Address': ['Unknown Address']
        })
        result = extract_district_city(df)
        
        assert result['District'].iloc[0] == 'Unknown'
        assert result['City'].iloc[0] == 'Unknown'

    def test_missing_address(self):
        """Test handling of missing address values."""
        df = pd.DataFrame({
            'Address': [None, 'Valid, Address']
        })
        result = extract_district_city(df)
        
        assert result['District'].iloc[0] == 'Unknown'
        assert result['City'].iloc[0] == 'Unknown'

    def test_vietnamese_addresses(self):
        """Test various Vietnamese address formats."""
        addresses = [
            '123 Nguyen Hue, Phuong 1, Quan 1, Ho Chi Minh',
            '45 Le Lai, Ha Bong, TP. Tan An, Long An',
            '78 Tran Phu, Phuong 2, Quan 5, Ho Chi Minh',
        ]
        df = pd.DataFrame({'Address': addresses})
        result = extract_district_city(df)
        
        assert len(result) == 3
        assert all(result['City'].notna())
        assert all(result['District'].notna())


class TestTargetEncode:
    """Tests for target encoding functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'District': ['Go Vap', 'Go Vap', 'Go Vap', 'Go Vap',
                        'Thu Duc', 'Thu Duc', 'Thu Duc',
                        'District 1', 'District 1'],
            'City': ['Ho Chi Minh', 'Ho Chi Minh', 'Ho Chi Minh', 'Ho Chi Minh',
                    'Ho Chi Minh', 'Ho Chi Minh', 'Ho Chi Minh',
                    'Ho Chi Minh', 'Ho Chi Minh'],
            'Price': [5.0, 6.0, 5.5, 5.2, 8.0, 8.5, 7.8, 10.0, 11.0]
        })

    def test_target_encode_with_smoothing(self, sample_data):
        """Test target encoding with smoothing returns correct structure."""
        encodings = target_encode_with_smoothing(sample_data.copy())
        
        assert 'district_encoding' in encodings
        assert 'city_encoding' in encodings
        assert 'district_global_mean' in encodings
        assert 'city_global_mean' in encodings
        
        # Check Go Vap encoding is computed
        assert 'Go Vap' in encodings['district_encoding']

    def test_target_encode_with_oof(self, sample_data):
        """Test out-of-fold target encoding."""
        df, encodings = target_encode_with_oof(sample_data.copy())
        
        assert 'District_Encoded' in df.columns
        assert 'City_Encoded' in df.columns
        
        # All districts should have encoded values
        assert df['District_Encoded'].notna().all()
        assert df['City_Encoded'].notna().all()

    def test_apply_target_encoding(self, sample_data):
        """Test applying target encoding from encodings dict."""
        encodings = target_encode_with_smoothing(sample_data.copy())
        df = apply_target_encoding(sample_data.copy(), encodings)
        
        assert 'District_Encoded' in df.columns
        assert 'City_Encoded' in df.columns
        
        # Go Vap should have encoded value
        go_vap_mask = df['District'] == 'Go Vap'
        assert df.loc[go_vap_mask, 'District_Encoded'].iloc[0] == encodings['district_encoding']['Go Vap']


class TestPreprocessData:
    """Tests for preprocess_data function."""

    @pytest.fixture
    def raw_data(self):
        """Create raw data similar to house_data.csv."""
        return pd.DataFrame({
            'Address': ['123 Nguyen Trai, Go Vap, Ho Chi Minh',
                       '456 Le Lai, District 1, Ho Chi Minh'],
            'Area': [80.0, 100.0],
            'Frontage': [5.0, 6.0],
            'Access Road': [6.0, 5.0],
            'House direction': ['Dong', 'Nam'],
            'Balcony direction': ['Nam', 'Tay'],
            'Floors': [3.0, 4.0],
            'Bedrooms': [3.0, 4.0],
            'Bathrooms': [2.0, 3.0],
            'Legal status': ['Have certificate', 'Sale contract'],
            'Furniture state': ['Full', 'Basic'],
            'Price': [5.5, 8.0]
        })

    def test_preprocess_removes_price_na(self):
        """Test that rows with missing Price are removed."""
        df = pd.DataFrame({
            'Address': ['Test, City'],
            'Area': [80],
            'Price': [None]
        })
        result, _ = preprocess_data(df)
        assert len(result) == 0

    def test_preprocess_extracts_location(self, raw_data):
        """Test that location features are extracted."""
        result, _ = preprocess_data(raw_data)
        
        assert 'District_Encoded' in result.columns
        assert 'City_Encoded' in result.columns
        
        # Original text columns should be dropped
        assert 'Address' not in result.columns
        assert 'District' not in result.columns
        assert 'City' not in result.columns

    def test_preprocess_one_hot_encoding(self, raw_data):
        """Test that categorical columns are one-hot encoded."""
        result, _ = preprocess_data(raw_data)
        
        # Should have one-hot encoded columns
        house_dir_cols = [c for c in result.columns if c.startswith('House direction_')]
        assert len(house_dir_cols) > 0
        
        furniture_cols = [c for c in result.columns if c.startswith('Furniture state_')]
        assert len(furniture_cols) > 0

    def test_preprocess_numeric_unchanged(self, raw_data):
        """Test that numeric columns are preserved."""
        result, _ = preprocess_data(raw_data)
        
        assert 'Area' in result.columns
        assert 'Frontage' in result.columns
        assert 'Floors' in result.columns
        assert 'Bedrooms' in result.columns
        assert 'Bathrooms' in result.columns
        assert 'Price' in result.columns

    def test_preprocess_output_shape(self, raw_data):
        """Test that output DataFrame has more columns than input."""
        result, _ = preprocess_data(raw_data)
        
        # Should have more columns due to one-hot encoding
        assert len(result.columns) > len(raw_data.columns)
    
    def test_preprocess_returns_encodings(self, raw_data):
        """Test that preprocess_data returns encodings dict."""
        _, encodings = preprocess_data(raw_data)
        
        assert encodings is not None
        assert 'district_encoding' in encodings
        assert 'city_encoding' in encodings


class TestIntegration:
    """Integration tests for the full preprocessing pipeline."""

    def test_preprocess_and_train_data_match(self):
        """Test that processed data matches what train.py expects."""
        # Load processed data
        processed_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'house_processed.csv')
        
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)
            
            # Check required columns exist
            required_numeric = ['Area', 'Frontage', 'Access Road', 'Floors', 'Bedrooms', 'Bathrooms', 'Price']
            for col in required_numeric:
                assert col in df.columns, f"Missing required column: {col}"
            
            # Check location encoding columns exist
            assert 'District_Encoded' in df.columns
            assert 'City_Encoded' in df.columns
            
            # Check no text location columns remain
            assert 'Address' not in df.columns
            assert 'District' not in df.columns
            assert 'City' not in df.columns
            
            # Check target column is present
            assert df['Price'].notna().all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
