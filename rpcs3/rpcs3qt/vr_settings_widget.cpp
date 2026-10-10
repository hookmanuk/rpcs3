#include "stdafx.h"
#include "vr_settings_widget.h"
#include "ui_vr_settings_widget.h"
#include "emu_settings.h"
#include "emu_settings_type.h"
#include "tooltips.h"

#include "Emu/GameInfo.h"
#include "Emu/System.h"
#include "Emu/RSX/Capture/rsx_camera_probe.h"

#include <QSignalBlocker>

vr_settings_widget::vr_settings_widget(QWidget* parent)
	: QWidget(parent), ui(new Ui::vr_settings_widget)
{
	ui->setupUi(this);
}

vr_settings_widget::~vr_settings_widget()
{
}

void vr_settings_widget::init(std::shared_ptr<emu_settings> emu_settings, const GameInfo* game, tooltip_subscriber subscribe_tooltip, slider_snapper snap_slider)
{
	m_emu_settings = std::move(emu_settings);
	const Tooltips tooltips;

	const auto enhance_checkbox = [&](emu_settings_type type, QCheckBox* checkbox, const QString& tooltip)
	{
		m_emu_settings->EnhanceCheckBox(checkbox, type);
		subscribe_tooltip(checkbox, tooltip);
	};

	// Only titles with a VR profile can be rendered in stereo, and the setting lives in
	// that game's custom configuration. Without a profile (including the global settings)
	// the VR section is hidden.
	const auto vr_profile = game ? rsx::vr::load_title_profile(game->serial) : nullptr;
	const bool vr_profiled_title = vr_profile != nullptr;
	setVisible(vr_profiled_title);
	enhance_checkbox(emu_settings_type::VREnabled, ui->vrEnabled, tooltips.settings.vr_enabled);
	enhance_checkbox(emu_settings_type::VRHudFixed, ui->vrHudFixed, tooltips.settings.vr_hud_fixed);
	enhance_checkbox(emu_settings_type::VRFixedScreen, ui->vrFixedScreen, tooltips.settings.vr_fixed_screen);
	m_emu_settings->EnhanceComboBox(ui->vrFrameRate, emu_settings_type::VRFrameRate);
	m_emu_settings->EnhanceComboBox(ui->vrCinematicScenes, emu_settings_type::VRCinematicScenes);
	subscribe_tooltip(ui->gb_vrCinematicScenes, tooltips.settings.vr_cinematic_scenes);
	const bool cinematic = vr_profile && vr_profile->reduced_scale_percent;
	ui->gb_vrCinematicScenes->setVisible(cinematic);
	// "Game": the options this game's VR profile offers (its rules tagged "option", cinematic scenes), first so they are
	// seen. Each shows only when the profile has it; the box hides when there are none. "General": every game's settings.
	{
		bool any = cinematic;
		const auto game_option = [&](QCheckBox* checkbox, emu_settings_type type, std::string_view option, const QString& tooltip)
		{
			enhance_checkbox(type, checkbox, tooltip);
			const bool offered = vr_profile && rsx::vr::profile_has_option(*vr_profile, option);
			checkbox->setVisible(offered);
			any |= offered;
		};
		game_option(ui->vrHighestDetailModels, emu_settings_type::VRHighestDetailModels, "highest_detail", tooltips.settings.vr_highest_detail_models);
		game_option(ui->vrReducedRateReflections, emu_settings_type::VRReducedRateReflections, "reflections", tooltips.settings.vr_reduced_rate_reflections);
		game_option(ui->vrReducedRateMirror, emu_settings_type::VRReducedRateMirror, "mirror", tooltips.settings.vr_reduced_rate_mirror);
		game_option(ui->vrSimplerDistantCars, emu_settings_type::VRSimplerDistantCars, "distant_cars", tooltips.settings.vr_simpler_distant_cars);
		ui->gb_vrGame->setVisible(any);
	}
	{
		// Each game's default rate, from its VR profile, with the headset refresh rates that
		// are an exact multiple of it.
		QString tooltip = tooltips.settings.vr_frame_rate;
		if (vr_profiled_title)
		{
			for (const auto& game_rate : rsx::vr::title_frame_rates(game->serial))
			{
				const QString name = QString::fromStdString(game_rate.name);
				if (!game_rate.default_fps)
				{
					tooltip += tr("\n%1: headset refresh rate (any)", "VR frame rate").arg(name);
					continue;
				}
				QStringList rates;
				for (const u32 hz : {60u, 72u, 75u, 80u, 90u, 100u, 120u, 144u})
				{
					if (hz % game_rate.default_fps == 0)
					{
						rates << QString::number(hz);
					}
				}
				const QString headset = rates.empty() ? QString() : tr(", headset %1 Hz", "VR frame rate").arg(rates.join("/"));
				if (game_rate.max_fps == game_rate.default_fps)
				{
					tooltip += tr("\n%1: %2 FPS (maximum%3)", "VR frame rate").arg(name).arg(game_rate.default_fps).arg(headset);
				}
				else if (headset.isEmpty())
				{
					tooltip += tr("\n%1: %2 FPS", "VR frame rate").arg(name).arg(game_rate.default_fps);
				}
				else
				{
					tooltip += tr("\n%1: %2 FPS (%3)", "VR frame rate").arg(name).arg(game_rate.default_fps).arg(headset.mid(2));
				}
			}
		}
		subscribe_tooltip(ui->gb_vrFrameRate, tooltip);
	}
	if (vr_profiled_title)
	{
		// Only rates up to the most any game of this title works at (a collection's games
		// share this configuration; each clamps to its own maximum when it runs).
		// Signals are blocked so nothing is saved here.
		const u32 max_fps = rsx::vr::title_max_fps(game->serial);
		const QSignalBlocker blocker(ui->vrFrameRate);
		const int current = ui->vrFrameRate->currentIndex();
		QStringList defaults;
		for (const u32 fps : rsx::vr::title_default_fps(game->serial))
		{
			defaults << (fps ? tr("%1 FPS", "VR frame rate").arg(fps) : tr("headset refresh rate", "VR frame rate"));
		}
		for (int i = ui->vrFrameRate->count() - 1; i >= 0; --i)
		{
			const QVariantList data = ui->vrFrameRate->itemData(i).toList();
			if (data.size() != 2)
			{
				continue;
			}
			if (data[1].toUInt() == 0)
			{
				// "Default (60 FPS)", or each game's for a collection ("Default (30 FPS / 60 FPS)").
				ui->vrFrameRate->setItemText(i, tr("Default (%1)", "VR frame rate").arg(defaults.join(" / ")));
			}
			else if (i != current && !rsx::vr::frame_rate_option_allowed(data[1].toUInt(), max_fps))
			{
				ui->vrFrameRate->removeItem(i);
			}
		}
	}
	{
		// Percentage sliders with a value label and a reset button.
		const auto percent_text = [](int value)
		{
			return tr("%1%", "VR HUD slider").arg(value);
		};
		const auto enhance_vr_slider = [&](QSlider* slider, QLabel* min, QLabel* max, QLabel* val, QAbstractButton* reset,
										   emu_settings_type type, QGroupBox* group, const QString& tooltip, std::function<QString(int)> format, int snap)
		{
			m_emu_settings->EnhanceSlider(slider, type);
			subscribe_tooltip(group, tooltip);
			const int def = std::stoi(m_emu_settings->GetSettingDefault(type));
			const auto text = [def, format](int value)
			{
				return value == def ? tr("%1 (Default)", "VR HUD slider").arg(format(value)) : format(value);
			};
			slider->setPageStep(10);
			min->setText(format(slider->minimum()));
			max->setText(format(slider->maximum()));
			val->setText(text(slider->value()));
			connect(slider, &QSlider::valueChanged, [text, val](int value)
				{
					val->setText(text(value));
				});
			connect(reset, &QAbstractButton::clicked, [def, slider]()
				{
					slider->setValue(def);
				});
			snap_slider(slider, snap);
		};

		enhance_vr_slider(ui->vrHudScale, ui->vrHudScaleMin, ui->vrHudScaleMax, ui->vrHudScaleVal, ui->vrHudScaleReset,
			emu_settings_type::VRHudScale, ui->gb_vrHudScale, tooltips.settings.vr_hud_scale, percent_text, 5);
		enhance_vr_slider(ui->vrHudOffsetX, ui->vrHudOffsetXMin, ui->vrHudOffsetXMax, ui->vrHudOffsetXVal, ui->vrHudOffsetXReset,
			emu_settings_type::VRHudOffsetX, ui->gb_vrHudOffsetX, tooltips.settings.vr_hud_offset, percent_text, 1);
		enhance_vr_slider(ui->vrHudOffsetY, ui->vrHudOffsetYMin, ui->vrHudOffsetYMax, ui->vrHudOffsetYVal, ui->vrHudOffsetYReset,
			emu_settings_type::VRHudOffsetY, ui->gb_vrHudOffsetY, tooltips.settings.vr_hud_offset, percent_text, 1);
		enhance_vr_slider(ui->vrScreenDepth, ui->vrScreenDepthMin, ui->vrScreenDepthMax, ui->vrScreenDepthVal, ui->vrScreenDepthReset,
			emu_settings_type::VRScreenDepth, ui->gb_vrScreenDepth, tooltips.settings.vr_screen_depth, percent_text, 5);
		const auto plain_text = [](int value)
		{
			return QString::number(value);
		};
		enhance_vr_slider(ui->vrCameraDepth, ui->vrCameraDepthMin, ui->vrCameraDepthMax, ui->vrCameraDepthVal, ui->vrCameraDepthReset,
			emu_settings_type::VRCameraDepth, ui->gb_vrCameraDepth, tooltips.settings.vr_camera_depth, plain_text, 5);
		enhance_vr_slider(ui->vrWorldScale, ui->vrWorldScaleMin, ui->vrWorldScaleMax, ui->vrWorldScaleVal, ui->vrWorldScaleReset,
			emu_settings_type::VRWorldScale, ui->gb_vrWorldScale, tooltips.settings.vr_world_scale, percent_text, 5);
		const auto degree_text = [](int value)
		{
			return tr("%1\u00b0", "VR slider").arg(value);
		};
		enhance_vr_slider(ui->vrReprojectionMargin, ui->vrReprojectionMarginMin, ui->vrReprojectionMarginMax, ui->vrReprojectionMarginVal, ui->vrReprojectionMarginReset, emu_settings_type::VRReprojectionMargin, ui->gb_vrReprojectionMargin, tooltips.settings.vr_reprojection_margin, [degree_text](int value)
			{
				return value < 0 ? tr("Auto", "VR reprojection margin") : degree_text(value);
			},
			1);

		const auto enable_vr_options = [this, vr_profiled_title]()
		{
			const bool vr = vr_profiled_title && ui->vrEnabled->isChecked();
			ui->vrHudFixed->setEnabled(vr);
			ui->gb_vrFrameRate->setEnabled(vr);
			ui->gb_vrGame->setEnabled(vr);
			ui->vrFixedScreen->setEnabled(vr);
			ui->gb_vrHudScale->setEnabled(vr);
			ui->gb_vrHudOffsetX->setEnabled(vr);
			ui->gb_vrHudOffsetY->setEnabled(vr);
			ui->gb_vrScreenDepth->setEnabled(vr && ui->vrFixedScreen->isChecked());
			ui->gb_vrCameraDepth->setEnabled(vr && !ui->vrFixedScreen->isChecked());
			ui->gb_vrWorldScale->setEnabled(vr && !ui->vrFixedScreen->isChecked());
			ui->gb_vrReprojectionMargin->setEnabled(vr && !ui->vrFixedScreen->isChecked());
		};
		connect(ui->vrEnabled, &QCheckBox::toggled, this, enable_vr_options);
		connect(ui->vrFixedScreen, &QCheckBox::toggled, this, enable_vr_options);
		enable_vr_options();

		// The headset session and right-eye resources are set up when the game boots,
		// so VR cannot be switched while one is running. The HUD options stay live.
		if (!Emu.IsStopped())
		{
			ui->vrEnabled->setEnabled(false);
			ui->vrEnabled->setText(tr("Enable VR Support (stop the game to change)"));
		}
	}
}
