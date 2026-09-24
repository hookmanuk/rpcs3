#include "stdafx.h"
#include "rpcs3_version.h"
#include "git-version.h"
#include "Utilities/StrUtil.h"

namespace rpcs3
{
	std::string_view get_branch()
	{
		return RPCS3_GIT_BRANCH;
	}

	std::string_view get_full_branch()
	{
		return RPCS3_GIT_FULL_BRANCH;
	}

	std::pair<std::string, std::string> get_commit_and_hash()
	{
		auto commit_and_hash = fmt::split(RPCS3_GIT_VERSION, {"-"});
		if (commit_and_hash.size() != 2)
			return std::make_pair("0", "00000000");

		return std::make_pair(std::move(commit_and_hash[0]), std::move(commit_and_hash[1]));
	}

	// VR fork: builds branched from upstream 0.0.42 are 0.0.42-vr1, -vr2, ... (GitHub release tags v0.0.42-vrN).
	// The tag leads the version postfix: "0.0.42-vr1-<commit> Alpha".
#define RPCS3_VR_VERSION "vr2"

	const utils::version& get_version()
	{
		static constexpr utils::version version{ 0, 0, 42, utils::version_type::alpha, 1, RPCS3_VR_VERSION "-" RPCS3_GIT_VERSION };
		return version;
	}

	std::string get_version_and_branch()
	{
		// VR fork: the VR branch shows its full version, "0.0.42-vr1-<commit> Alpha" (branch and build
		// details stay in the log).
		if (rpcs3::get_branch() == "openxr"sv)
		{
			return rpcs3::get_version().to_string();
		}

		// Add branch and commit hash to version on frame unless it's master.
		if (rpcs3::get_branch() != "master"sv && rpcs3::get_branch() != "HEAD"sv)
		{
			return get_verbose_version();
		}

		// Get version by substringing VersionNumber-buildnumber-commithash to get just the part before the dash
		std::string version = rpcs3::get_version().to_string();

		const auto last_minus = version.find_last_of('-');
		version = version.substr(0, last_minus);

		return version;
	}

	std::string get_verbose_version()
	{
		std::string version = fmt::format("%s | %s", rpcs3::get_version().to_string(), get_branch());
		if (is_local_build())
		{
			fmt::append(version, " | local_build");
		}
		return version;
	}

	bool is_release_build()
	{
		static constexpr bool is_release_build = std::string_view(RPCS3_GIT_FULL_BRANCH) == "RPCS3/rpcs3/master"sv;
		return is_release_build;
	}

	bool is_local_build()
	{
		static constexpr bool is_local_build = std::string_view(RPCS3_GIT_FULL_BRANCH) == "local_build"sv;
		return is_local_build;
	}
}
