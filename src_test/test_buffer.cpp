#include "TestHelper.h"
#include "test.h"

#include <cassert>
#include <memory>

namespace
{
void test_invalid_buffer_size(LLGI::DeviceType deviceType)
{
	LLGI::PlatformParameter platformParameter;
	platformParameter.Device = deviceType;

	auto window = std::unique_ptr<LLGI::Window>(LLGI::CreateWindow("InvalidBufferSize", LLGI::Vec2I(320, 240)));
	auto platform = LLGI::CreateSharedPtr(LLGI::CreatePlatform(platformParameter, window.get()));
	assert(platform != nullptr);

	auto graphics = LLGI::CreateSharedPtr(platform->CreateGraphics());
	assert(graphics != nullptr);
	assert(graphics->CreateBuffer(LLGI::BufferUsageType::Vertex, 0) == nullptr);
	assert(graphics->CreateBuffer(LLGI::BufferUsageType::Vertex, -1) == nullptr);
}
} // namespace

TestRegister Buffer_InvalidSize("Buffer.InvalidSize", [](LLGI::DeviceType device) -> void { test_invalid_buffer_size(device); });
