#pragma once

// VR fork: the VR group box of the settings dialog's GPU tab, in its own widget so that
// settings_dialog.ui and settings_dialog.cpp carry a placeholder and one call.

#include <QWidget>

#include <functional>
#include <memory>

class emu_settings;
class QSlider;
struct GameInfo;

namespace Ui
{
	class vr_settings_widget;
}

class vr_settings_widget : public QWidget
{
	Q_OBJECT

public:
	// The dialog's tooltip subscription (settings_dialog::SubscribeTooltip) and slider snapping (SnapSlider).
	using tooltip_subscriber = std::function<void(QObject*, const QString&)>;
	using slider_snapper = std::function<void(QSlider*, int)>;

	explicit vr_settings_widget(QWidget* parent = nullptr);
	~vr_settings_widget();

	void init(std::shared_ptr<emu_settings> emu_settings, const GameInfo* game, tooltip_subscriber subscribe_tooltip, slider_snapper snap_slider);

private:
	std::unique_ptr<Ui::vr_settings_widget> ui;
	std::shared_ptr<emu_settings> m_emu_settings;
};
