// tests/ml/test_iou_tracker.cpp
#include <gtest/gtest.h>
#include "improc/ml/tracking/track.hpp"

using namespace improc::ml;

TEST(TrackTest, DefaultConstruct) {
    Track t;
    EXPECT_EQ(t.id, -1);
    EXPECT_FALSE(t.is_confirmed);
}
