// Build-only compatibility: Windows FILE_CREATE macro conflicts with the
// existing HighFive PropertyType::FILE_CREATE enum. Preserve upstream sources.
#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef FILE_CREATE
#undef FILE_CREATE
#endif
#endif
