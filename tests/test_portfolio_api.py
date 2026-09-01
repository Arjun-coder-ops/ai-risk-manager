"""Tests for portfolio API endpoints (single-date and date-range)."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

import pytest
from app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


class TestPortfolioAPISingleDate:
    """Tests for single-date portfolio endpoint."""
    
    def test_single_date_query_works(self, client):
        """GET /api/risk/portfolio?date=... returns 200 with merchants."""
        response = client.get('/api/risk/portfolio?date=2026-05-31')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        assert data['date'] == '2026-05-31'
        assert 'merchants' in data
        assert len(data['merchants']) == 150
    
    def test_single_date_response_structure(self, client):
        """Single-date response has expected fields per merchant."""
        response = client.get('/api/risk/portfolio?date=2026-05-31')
        data = response.get_json()
        assert len(data['merchants']) > 0
        
        first_merchant = data['merchants'][0]
        assert 'merchant_id' in first_merchant
        assert 'date' in first_merchant
        assert 'risk_score' in first_merchant
        assert 'risk_level' in first_merchant
        assert 'status' in first_merchant
        assert 'signals' in first_merchant
    
    def test_no_date_params_errors(self, client):
        """GET /api/risk/portfolio with no params returns 400."""
        response = client.get('/api/risk/portfolio')
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'
    
    def test_ambiguous_params_errors(self, client):
        """GET /api/risk/portfolio?date=...&from_date=... returns 400."""
        response = client.get('/api/risk/portfolio?date=2026-05-31&from_date=2026-05-01&to_date=2026-05-31')
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'
        assert 'both' in data['reason'].lower() or 'not both' in data['reason'].lower()


class TestPortfolioAPIDateRange:
    """Tests for date-range portfolio endpoint."""
    
    def test_daterange_query_works(self, client):
        """GET /api/risk/portfolio?from_date=...&to_date=... returns 200."""
        response = client.get('/api/risk/portfolio?from_date=2026-05-29&to_date=2026-05-31')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        assert 'date_range' in data
        assert 'summary' in data
        assert 'dates' in data
    
    def test_daterange_response_structure(self, client):
        """Date-range response has expected fields."""
        response = client.get('/api/risk/portfolio?from_date=2026-05-29&to_date=2026-05-31')
        data = response.get_json()
        
        # Check date_range
        assert data['date_range']['from'] == '2026-05-29'
        assert data['date_range']['to'] == '2026-05-31'
        
        # Check summary
        assert 'dates_scored' in data['summary']
        assert 'total_merchant_days' in data['summary']
        assert 'high_risk_total' in data['summary']
        assert 'review_risk_total' in data['summary']
        assert 'low_risk_total' in data['summary']
        
        # Check dates structure
        assert len(data['dates']) > 0
        for date_entry in data['dates']:
            assert 'date' in date_entry
            assert 'total_monitored' in date_entry
            assert 'high_count' in date_entry
            assert 'review_count' in date_entry
            assert 'low_count' in date_entry
            assert 'top_merchants' in date_entry
    
    def test_daterange_top_merchants_optimized(self, client):
        """Date-range top_merchants have essential fields, not full signals."""
        response = client.get('/api/risk/portfolio?from_date=2026-05-30&to_date=2026-05-31')
        data = response.get_json()
        
        # Get first date's top merchants
        if data['dates'] and data['dates'][0]['top_merchants']:
            top_merchant = data['dates'][0]['top_merchants'][0]
            
            # Should have essential fields
            assert 'merchant_id' in top_merchant
            assert 'risk_score' in top_merchant
            assert 'risk_level' in top_merchant
            assert 'recommended_action' in top_merchant
            
            # Should have top_signal (but not full signals array)
            assert 'top_signal' in top_merchant
            
            # Should NOT have full signals (that's for single-date view)
            assert 'signals' not in top_merchant or isinstance(top_merchant.get('signals'), dict)
    
    def test_daterange_chronological_order(self, client):
        """Dates in range response are in chronological order."""
        response = client.get('/api/risk/portfolio?from_date=2026-05-28&to_date=2026-05-31')
        data = response.get_json()
        
        dates = [d['date'] for d in data['dates']]
        assert dates == sorted(dates), "Dates should be chronological"
    
    def test_daterange_summary_counts_correct(self, client):
        """Summary counts match aggregated date counts."""
        response = client.get('/api/risk/portfolio?from_date=2026-05-30&to_date=2026-05-31')
        data = response.get_json()
        
        summary = data['summary']
        dates = data['dates']
        
        # Calculate from dates
        high_total = sum(d['high_count'] for d in dates)
        review_total = sum(d['review_count'] for d in dates)
        low_total = sum(d['low_count'] for d in dates)
        merchant_days_total = sum(d['total_monitored'] for d in dates)
        
        assert summary['high_risk_total'] == high_total
        assert summary['review_risk_total'] == review_total
        assert summary['low_risk_total'] == low_total
        assert summary['total_merchant_days'] == merchant_days_total
        assert summary['dates_scored'] == len(dates)
    
    def test_invalid_date_range_errors(self, client):
        """GET /api/risk/portfolio?from_date > to_date returns 400."""
        response = client.get('/api/risk/portfolio?from_date=2026-05-31&to_date=2026-05-01')
        assert response.status_code == 400
        data = response.get_json()
        assert 'cannot be after' in data['reason'].lower()
    
    def test_missing_range_params_errors(self, client):
        """GET /api/risk/portfolio?from_date=... (missing to_date) returns 400."""
        response = client.get('/api/risk/portfolio?from_date=2026-05-01')
        assert response.status_code == 400
        data = response.get_json()
        assert data['status'] == 'error'
    
    def test_excessive_range_errors(self, client):
        """Date range > 60 days returns 400."""
        response = client.get('/api/risk/portfolio?from_date=2026-01-01&to_date=2026-06-29')  # 180 days
        assert response.status_code == 400
        data = response.get_json()
        assert '60 days' in data['reason']
    
    def test_max_allowed_range_works(self, client):
        """Date range of exactly 60 days is allowed."""
        # 2026-05-01 to 2026-06-29 is ~60 days
        response = client.get('/api/risk/portfolio?from_date=2026-04-30&to_date=2026-06-28')
        # Should not return 400 (may return 200 or 400 for other reasons, but not range error)
        if response.status_code == 400:
            data = response.get_json()
            assert '60 days' not in data['reason']
    
    def test_daterange_outside_dataset(self, client):
        """Date range outside dataset returns gracefully with empty results."""
        response = client.get('/api/risk/portfolio?from_date=2026-07-01&to_date=2026-07-10')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        # Dates outside dataset should result in empty dates array or note
        assert 'dates' in data
    
    def test_daterange_partial_overlap(self, client):
        """Date range partially outside dataset returns only available dates."""
        response = client.get('/api/risk/portfolio?from_date=2026-06-28&to_date=2026-07-10')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'ok'
        # Should have data for 2026-06-28 and 2026-06-29 only
        dates = [d['date'] for d in data['dates']]
        assert all(d <= '2026-06-29' for d in dates)


class TestPortfolioAPIBackwardCompat:
    """Tests to ensure backward compatibility."""
    
    def test_single_date_format_unchanged(self, client):
        """Single-date response format matches original implementation."""
        response = client.get('/api/risk/portfolio?date=2026-05-31')
        data = response.get_json()
        
        # Original format
        assert 'status' in data
        assert 'date' in data
        assert 'merchants' in data
        
        # Merchants array format
        merchants = data['merchants']
        assert isinstance(merchants, list)
        if len(merchants) > 0:
            m = merchants[0]
            assert isinstance(m, dict)
            assert all(k in m for k in ['status', 'merchant_id', 'date', 'risk_score', 'risk_level'])
    
    def test_single_date_merchant_scores_consistent(self, client):
        """Merchant scores are consistent on repeated calls."""
        response1 = client.get('/api/risk/portfolio?date=2026-05-31')
        response2 = client.get('/api/risk/portfolio?date=2026-05-31')
        
        data1 = response1.get_json()
        data2 = response2.get_json()
        
        # Same merchants, same order, same scores
        assert data1['merchants'][0]['merchant_id'] == data2['merchants'][0]['merchant_id']
        assert data1['merchants'][0]['risk_score'] == data2['merchants'][0]['risk_score']
    
    def test_single_date_vs_range_consistency(self, client):
        """A merchant's score in single-date matches score in range result."""
        single_response = client.get('/api/risk/portfolio?date=2026-05-31')
        range_response = client.get('/api/risk/portfolio?from_date=2026-05-31&to_date=2026-05-31')
        
        single_data = single_response.get_json()
        range_data = range_response.get_json()
        
        # Both should have M0083 (a merchant in the data)
        single_merchants = {m['merchant_id']: m for m in single_data['merchants']}
        range_merchants = {d['top_merchants'][0]['merchant_id']: d['top_merchants'][0] 
                          for d in range_data['dates'] if d['top_merchants']}
        
        # At least one merchant should be in both
        common_merchants = set(single_merchants.keys()) & set(range_merchants.keys())
        if common_merchants:
            m_id = list(common_merchants)[0]
            single_score = single_merchants[m_id]['risk_score']
            range_score = range_merchants[m_id]['risk_score']
            assert single_score == range_score or abs(single_score - range_score) < 0.001
