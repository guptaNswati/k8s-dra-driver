/*
 * Copyright (c) 2025 NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package workqueue

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestWorkQueue(t *testing.T) {
	// Create a WorkQueue with the default rate limiter
	defaultRateLimiter := DefaultControllerRateLimiter()
	wq := New(defaultRateLimiter)
	require.NotNil(t, wq)
	require.NotNil(t, wq.queue)

	// Create a context with timeout for processing.
	// Use a longer timeout to ensure processing completes.
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Define test cases
	tests := []struct {
		name     string
		obj      any
		callback func(context.Context, any) error
		validate func(t *testing.T, called *int32)
	}{
		{
			name: "EnqueueRaw",
			obj:  "AnyObject",
			callback: func(ctx context.Context, obj any) error {
				return nil
			},
			validate: func(t *testing.T, called *int32) {
				if atomic.LoadInt32(called) != 1 {
					t.Error("EnqueueRaw callback was not invoked")
				}
			},
		},
		{
			name: "EnqueueValid",
			obj:  &runtime.Unknown{},
			callback: func(ctx context.Context, obj any) error {
				if _, ok := obj.(runtime.Object); !ok {
					t.Errorf("Expected runtime.Object, got %T", obj)
				}
				return nil
			},
			validate: func(t *testing.T, called *int32) {
				if atomic.LoadInt32(called) != 1 {
					t.Error("Enqueue callback was not invoked")
				}
			},
		},
		{
			name:     "EnqueueInvalid",
			obj:      "NotRuntimeObject",
			callback: func(ctx context.Context, obj any) error { return nil },
			validate: func(t *testing.T, called *int32) {
				// No validation needed for invalid objects
			},
		},
		{
			name: "NilCallback",
			obj:  &runtime.Unknown{},
			validate: func(t *testing.T, called *int32) {
				// No validation needed for nil callbacks
			},
		},
	}

	// Run test cases
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var called int32

			// Wrap the callback to track invocations
			wrappedCallback := func(ctx context.Context, obj any) error {
				if tt.callback != nil {
					err := tt.callback(ctx, obj)
					if err != nil {
						return err
					}
				}
				atomic.StoreInt32(&called, 1)
				return nil
			}

			// Enqueue the item
			if tt.name == "NilCallback" {
				wq.Enqueue(tt.obj, nil)
			} else {
				wq.Enqueue(tt.obj, wrappedCallback)
			}

			// Process the item if it's not invalid
			if tt.name != "EnqueueInvalid" {
				wq.processNextWorkItem(ctx)
			}

			// Validate the test case
			if tt.validate != nil {
				tt.validate(t, &called)
			}
		})
	}

}
